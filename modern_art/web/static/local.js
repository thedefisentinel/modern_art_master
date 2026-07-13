/* 本地"后端":在浏览器里跑整局(引擎 + AI + 逐帧),对齐 server.py 的 payload/帧结构。
   前端只需调用 MALocal.newGame(...) / MALocal.action(...),不再需要服务器。 */
(function (root) {
'use strict';
const E = root.MAEngine, Ag = root.MAAgents;
let SESSION = null;                 // ONNX 会话(rl 用),懒加载
let G = null;                       // 当前对局:{state, seats, human}

async function ensureSession() {
  if (SESSION) return SESSION;
  SESSION = await root.ort.InferenceSession.create('policy.onnx');
  return SESSION;
}

function buildPayload(state, human) {
  const obs = E.observation(state, human);
  const yourTurn = !E.isOver(state) && E.currentPlayer(state) === human;
  obs.is_your_turn = yourTurn;
  const bb = E.bidBounds(state);
  obs.bid_bounds = (yourTurn && bb) ? bb : null;
  obs.log = state.log;
  obs.over = E.isOver(state);
  obs.scores = E.isOver(state) ? E.finalScores(state) : null;
  obs.winner = E.isOver(state) ? E.winner(state) : null;
  return obs;
}

const WON_RE = /→ P(\d+) 以 (\d+) 得/;
const FREE_RE = /→ P(\d+) 免费获得/;
function cardsOf(a) { if (!a) return []; if (a.cards.length) return a.cards.map(c=>({...c})); return a.double_card ? [{...a.double_card}] : []; }

function describe(before, after, action, actor) {
  const kind = action.kind;
  const newlog = after.log.slice(before.log.length);
  const ev = { actor, kind, log: newlog };
  if (after.round !== before.round || (E.isOver(after) && !E.isOver(before))) {
    ev.type = 'settle'; ev.settled_round = before.round;
    ev.deltas = after.players.map((p,i)=> p.money - before.players[i].money);
    ev.over = E.isOver(after); return ev;
  }
  const ab = before.auction, aa = after.auction;
  if (ab && !aa) {
    let won=null, free=null;
    for (const l of newlog) { const m=l.match(WON_RE); if (m) { won=m; break; } }
    if (!won) for (const l of newlog) { const m=l.match(FREE_RE); if (m) { free=m; break; } }
    if (won) { ev.type='won'; ev.winner=+won[1]; ev.price=+won[2]; }
    else if (free) { ev.type='won'; ev.winner=+free[1]; ev.price=0; ev.free=true; }
    else ev.type='nosale';
    ev.artist = ab.artist; ev.cards = cardsOf(ab); return ev;
  }
  if ((kind==='ChooseCard'||kind==='AddSecond') && aa) {
    ev.type='choose'; ev.seller=aa.seller; ev.artist=aa.artist; ev.auction_type=aa.auction_type; ev.cards=cardsOf(aa); return ev;
  }
  if (kind==='Bid') { ev.type='bid'; ev.player=actor; ev.amount=action.amount; }
  else if (kind==='Buy') { ev.type='buy'; ev.player=actor; ev.amount=ab?ab.high:null; }
  else if (kind==='SealedBid') { ev.type='sealed'; ev.player=actor; }
  else if (kind==='PassBid'||kind==='PassBuy') { ev.type='pass'; ev.player=actor; }
  else if (kind==='DeclineAdd') { ev.type='decline'; ev.player=actor; }
  else ev.type='other';
  return ev;
}

async function driveAiFrames(firstAction, firstActor) {
  const human = G.human, frames = [];
  const record = async (before, action, actor) => {
    G.state = E.apply(before, action);
    frames.push({ event: describe(before, G.state, action, actor), payload: buildPayload(G.state, human) });
  };
  await record(G.state, firstAction, firstActor);
  let guard = 0;
  while (!E.isOver(G.state) && E.currentPlayer(G.state) !== human) {
    const before = G.state, actor = E.currentPlayer(before);
    const action = await G.seats[actor].act(before);
    await record(before, action, actor);
    if (++guard > 10000) break;
  }
  return frames;
}

async function driveAiSilent() {   // 开局:人类非首位时静默推进
  while (!E.isOver(G.state) && E.currentPlayer(G.state) !== G.human) {
    const actor = E.currentPlayer(G.state);
    G.state = E.apply(G.state, await G.seats[actor].act(G.state));
  }
}

async function newGame(numPlayers, ai, humanSeat, seed) {
  humanSeat = humanSeat || 0;
  let session = null;
  if (ai === 'rl') session = await ensureSession();
  const state = E.newGame(numPlayers, seed);
  const seats = [];
  for (let i=0;i<numPlayers;i++) seats.push(i===humanSeat ? null : Ag.makeAgent(ai, (seed==null?null:seed+i+1), session));
  G = { state, seats, human: humanSeat };
  await driveAiSilent();
  const p = buildPayload(G.state, G.human);
  p.game_id = 'local';
  return p;
}

async function action(act) {
  if (!G || E.isOver(G.state)) return { error: '对局已结束' };
  if (E.currentPlayer(G.state) !== G.human) return { error: '现在不是你的回合' };
  let ok = true;
  try { E.apply(G.state, act); } catch (e) { return { error: String(e.message||e) }; }
  const frames = await driveAiFrames(act, G.human);
  return { frames, final: buildPayload(G.state, G.human) };
}

const API = { newGame, action, ensureSession };
if (typeof module!=='undefined' && module.exports) module.exports = API;
root.MALocal = API;
})(typeof window!=='undefined' ? window : globalThis);
