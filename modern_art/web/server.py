"""标准库 HTTP 服务器:把引擎暴露为 JSON API,并托管中文单页界面。

API:
    POST /api/new     {num_players, human_seat, ai}      -> {game_id, payload}
    GET  /api/state?game_id=..                            -> payload
    POST /api/action  {game_id, action:{kind,...}}        -> payload  (含自动驱动 AI 后的结果)

payload = observation(state, human_seat) 追加:
    is_your_turn, bid_bounds([min,max]|null), log(全量), over, scores, winner
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from ..engine import game
from ..engine.actions import from_dict as action_from_dict
from ..agents import AGENT_REGISTRY, make_agent

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 内存对局表:game_id -> {"state": GameState, "seats": [Agent|None], "human": int}
GAMES: dict[str, dict] = {}


# ── 载荷构造 / AI 驱动 ─────────────────────────────────────────────────────

def build_payload_state(state, human: int) -> dict:
    obs = game.observation(state, human)
    your_turn = (not game.is_over(state)) and game.current_player(state) == human
    obs["is_your_turn"] = your_turn
    obs["bid_bounds"] = list(game.bid_bounds(state)) if your_turn and game.bid_bounds(state) else None
    obs["log"] = state.log
    obs["over"] = game.is_over(state)
    obs["scores"] = game.final_scores(state) if game.is_over(state) else None
    obs["winner"] = game.winner(state) if game.is_over(state) else None
    return obs


def build_payload(sess: dict) -> dict:
    return build_payload_state(sess["state"], sess["human"])


_WON_RE = re.compile(r"→ P(\d+) 以 (\d+) 得")
_FREE_RE = re.compile(r"→ P(\d+) 免费获得")


def _cards_of(a) -> list:
    if a is None:
        return []
    if a.cards:
        return [c.to_dict() for c in a.cards]
    return [a.double_card.to_dict()] if a.double_card else []


def describe_transition(before, after, action, actor: int) -> dict:
    """比对一次 apply 前后的状态,给前端一个可播放的事件描述。"""
    kind = type(action).__name__
    newlog = after.log[len(before.log):]
    ev = {"actor": actor, "kind": kind, "log": newlog}

    # 回合结算 / 终局
    if after.round != before.round or (game.is_over(after) and not game.is_over(before)):
        ev["type"] = "settle"
        ev["settled_round"] = before.round
        ev["deltas"] = [after.players[i].money - before.players[i].money
                        for i in range(after.num_players)]
        ev["over"] = game.is_over(after)
        return ev

    a_b, a_a = before.auction, after.auction

    # 拍卖成交(拍卖从"有"变"无")
    if a_b is not None and a_a is None:
        won = next((_WON_RE.search(l) for l in newlog if _WON_RE.search(l)), None)
        free = next((_FREE_RE.search(l) for l in newlog if _FREE_RE.search(l)), None)
        if won:
            ev.update(type="won", winner=int(won.group(1)), price=int(won.group(2)))
        elif free:
            ev.update(type="won", winner=int(free.group(1)), price=0, free=True)
        else:
            ev["type"] = "nosale"
        ev["artist"] = a_b.artist.value
        ev["cards"] = _cards_of(a_b)
        return ev

    # 新拍卖开始(有人上架 / 补牌成局)
    if kind in ("ChooseCard", "AddSecond") and a_a is not None:
        ev.update(type="choose", seller=a_a.seller, artist=a_a.artist.value,
                  auction_type=a_a.auction_type.value, cards=_cards_of(a_a))
        return ev

    # 竞价过程
    if kind == "Bid":
        ev.update(type="bid", player=actor, amount=action.amount)
    elif kind == "Buy":
        ev.update(type="buy", player=actor, amount=(a_b.high if a_b else None))
    elif kind == "SealedBid":
        ev.update(type="sealed", player=actor)          # 不泄露金额
    elif kind in ("PassBid", "PassBuy"):
        ev.update(type="pass", player=actor)
    elif kind == "DeclineAdd":
        ev.update(type="decline", player=actor)
    else:
        ev["type"] = "other"
    return ev


def drive_ai_frames(sess: dict, first_action, first_actor: int) -> list:
    """执行人类动作 + 随后所有 AI 动作,逐步返回帧序列(每步一帧,含事件与状态快照)。"""
    human = sess["human"]
    frames = []

    def record(before, action, actor):
        sess["state"] = game.apply(before, action)
        frames.append({
            "event": describe_transition(before, sess["state"], action, actor),
            "payload": build_payload_state(sess["state"], human),
        })

    record(sess["state"], first_action, first_actor)      # 人类这一步
    guard = 0
    while not game.is_over(sess["state"]) and game.current_player(sess["state"]) != human:
        before = sess["state"]
        actor = game.current_player(before)
        record(before, sess["seats"][actor].act(before), actor)
        guard += 1
        if guard > 10000:
            break
    return frames


def drive_ai(sess: dict) -> None:
    """(用于开局:人类不是首位时)静默把 AI 走到人类回合。"""
    while not game.is_over(sess["state"]) and game.current_player(sess["state"]) != sess["human"]:
        actor = game.current_player(sess["state"])
        sess["state"] = game.apply(sess["state"], sess["seats"][actor].act(sess["state"]))


# ── HTTP 处理 ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ModernArt/0.1"

    def log_message(self, *args):  # 静音默认访问日志
        pass

    # -- 工具 --
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- 路由 --
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            qs = parse_qs(parsed.query)
            gid = (qs.get("game_id") or [""])[0]
            sess = GAMES.get(gid)
            if not sess:
                self._send_json({"error": "对局不存在"}, 404)
                return
            self._send_json(build_payload(sess))
            return
        # 其它路径:从 STATIC_DIR 直接当静态文件服务(engine.js / policy.onnx / art/…)。
        # 这样纯前端版的相对路径既能被本服务器服务,也能原样部署到静态托管。
        rel = os.path.normpath(route.lstrip("/")).replace("\\", "/")
        rel = rel[len("static/"):] if rel.startswith("static/") else rel  # 兼容旧的 /static/ 前缀
        if rel.startswith("..") or rel.startswith("/") or rel == "":
            self.send_error(403, "Forbidden")
            return
        fpath = os.path.join(STATIC_DIR, *rel.split("/"))
        if os.path.isfile(fpath):
            self._send_file(fpath, _guess_type(rel))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            data = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是合法 JSON"}, 400)
            return

        if route == "/api/new":
            self._handle_new(data)
        elif route == "/api/action":
            self._handle_action(data)
        else:
            self.send_error(404, "Not Found")

    # -- 业务 --
    def _handle_new(self, data: dict):
        num = int(data.get("num_players", 3))
        human = int(data.get("human_seat", 0))
        ai_kind = str(data.get("ai", "standard"))
        if ai_kind not in AGENT_REGISTRY and ai_kind != "rl":
            self._send_json({"error": f"未知 AI 类型 {ai_kind}"}, 400)
            return
        if not (3 <= num <= 5) or not (0 <= human < num):
            self._send_json({"error": "参数非法(人数 3-5,human_seat 越界)"}, 400)
            return
        seed = data.get("seed")
        state = game.new_game(num, seed=seed)
        try:
            seats = [
                None if i == human else make_agent(
                    ai_kind, seed=None if seed is None else int(seed) + i + 1)
                for i in range(num)
            ]
        except (FileNotFoundError, ValueError) as e:
            self._send_json({"error": str(e)}, 400)
            return
        gid = secrets.token_hex(8)
        sess = {"state": state, "seats": seats, "human": human}
        GAMES[gid] = sess
        drive_ai(sess)  # 若人类不是首位,先让 AI 走到人类回合
        payload = build_payload(sess)
        payload["game_id"] = gid
        self._send_json(payload)

    def _handle_action(self, data: dict):
        gid = str(data.get("game_id", ""))
        sess = GAMES.get(gid)
        if not sess:
            self._send_json({"error": "对局不存在"}, 404)
            return
        state = sess["state"]
        if game.is_over(state):
            self._send_json({"error": "对局已结束"}, 400)
            return
        if game.current_player(state) != sess["human"]:
            self._send_json({"error": "现在不是你的回合"}, 400)
            return
        try:
            action = action_from_dict(data["action"])
        except (KeyError, TypeError) as e:
            self._send_json({"error": f"动作格式错误:{e}"}, 400)
            return
        # 校验人类动作合法(先用克隆试跑,避免污染对局)
        try:
            game.apply(state, action)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return
        # 逐帧执行:人类这一步 + 随后所有 AI 步(供前端一步步动画播放)
        frames = drive_ai_frames(sess, action, sess["human"])
        self._send_json({"frames": frames, "final": build_payload(sess)})


def _guess_type(fname: str) -> str:
    if fname.endswith(".js"):
        return "application/javascript; charset=utf-8"
    if fname.endswith(".css"):
        return "text/css; charset=utf-8"
    if fname.endswith(".html"):
        return "text/html; charset=utf-8"
    return "application/octet-stream"


def main(argv=None):
    ap = argparse.ArgumentParser(description="现代艺术 网页服务器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"现代艺术 已启动:http://{args.host}:{args.port}  (Ctrl+C 停止)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
