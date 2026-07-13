/* 现代艺术 · 引擎 + 观测编码(JS 移植版,对齐 Python engine/)。
   纯静态:整局在浏览器里跑。状态是普通对象,形状与 Python state.to_dict() 一致,
   便于与 Python 对拍验证。同时可在 Node 里 require 做对拍测试。            */
(function (root) {
'use strict';

// ── 常量(对齐 rules.py)──────────────────────────────────────────────
const ARTISTS = ['lite_metal', 'yoko', 'christin_p', 'karl_gitter', 'krypto']; // 顺序=平局优先级
const TYPES = ['open', 'once', 'sealed', 'fixed', 'double'];                   // 顺序=AuctionType 枚举
const AIDX = {}; ARTISTS.forEach((a, i) => AIDX[a] = i);
const TIDX = {}; TYPES.forEach((t, i) => TIDX[t] = i);
const DIST = {
  lite_metal:  {open:3, once:3, sealed:2, fixed:2, double:2},
  yoko:        {open:3, once:2, sealed:3, fixed:3, double:2},
  christin_p:  {open:3, once:3, sealed:3, fixed:3, double:2},
  karl_gitter: {open:3, once:3, sealed:3, fixed:3, double:3},
  krypto:      {open:4, once:3, sealed:3, fixed:3, double:3},
};
const DEALT = {3:{1:10,2:6,3:6,4:0}, 4:{1:9,2:4,3:4,4:0}, 5:{1:8,2:3,3:3,4:0}};
const STARTING_MONEY = 100, NUM_ROUNDS = 4, RANK_VALUES = [30,20,10], TRIGGER = 5;
const NA = 5, NT = 5, MAXP = 5;

const clone = (s) => structuredClone(s);
const rotation = (n, start) => Array.from({length:n}, (_,k) => (start+k)%n);

// ── 建局 / 发牌 ─────────────────────────────────────────────────────
function buildDeck() {
  const deck = [];
  for (const artist of ARTISTS) {
    let aid = 0;
    for (const t of TYPES) for (let i=0;i<DIST[artist][t];i++) deck.push({artist, auction:t, art_id:aid++});
  }
  return deck;
}
function mulberry32(seed) { let a = seed>>>0; return () => { a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function shuffle(arr, rnd) { for (let i=arr.length-1;i>0;i--){ const j=Math.floor(rnd()*(i+1)); [arr[i],arr[j]]=[arr[j],arr[i]]; } }

function deal(s, rnd) {
  const count = DEALT[s.num_players][rnd];
  for (let i=0;i<count;i++) for (let p=0;p<s.num_players;p++) if (s.deck.length) s.players[p].hand.push(s.deck.pop());
}

function newGame(numPlayers, seed) {
  const deck = buildDeck();
  shuffle(deck, mulberry32(seed==null ? (Math.random()*1e9|0) : seed));
  const s = {
    num_players: numPlayers, round: 1, phase: 'choose',
    players: Array.from({length:numPlayers}, () => ({hand:[], money:STARTING_MONEY, purchases:[], paid_total:0})),
    deck,
    value_board: Object.fromEntries(ARTISTS.map(a=>[a,0])),
    value_markers: Object.fromEntries(ARTISTS.map(a=>[a,[0,0,0,0]])),
    round_counts: Object.fromEntries(ARTISTS.map(a=>[a,0])),
    active_player: 0, start_player: 0, auction: null, log: [],
  };
  deal(s, 1);
  openRound(s);
  s.log.push(`开局:${numPlayers} 人,每人初始资金 ${STARTING_MONEY}。第 1 回合开始。`);
  return s;
}

// ── 查询 ────────────────────────────────────────────────────────────
const isOver = (s) => s.phase === 'game_over';
function currentPlayer(s) {
  if (s.phase === 'choose') return s.active_player;
  if (s.phase === 'game_over') return -1;
  return s.auction.to_act;
}
const finalScores = (s) => s.players.map(p=>p.money);
function winner(s) { if (!isOver(s)) return []; const sc=finalScores(s); const b=Math.max(...sc); return sc.map((x,i)=>x===b?i:-1).filter(i=>i>=0); }
function bidBounds(s) {
  const a=s.auction; if(!a) return null; const p=currentPlayer(s); const money=s.players[p].money;
  if (s.phase==='bid_open'||s.phase==='bid_once') return [a.high+1, money];
  if (s.phase==='bid_sealed') return [0, money];
  return null;
}

// ── 合法动作 ────────────────────────────────────────────────────────
function legalActions(s) {
  const ph=s.phase;
  if (ph==='choose') {
    const hand=s.players[s.active_player].hand, acts=[];
    hand.forEach((c,i)=> acts.push(c.auction==='fixed' ? {kind:'ChooseCard',card_index:i,price:null} : {kind:'ChooseCard',card_index:i}));
    return acts;
  }
  if (ph==='double_offer') {
    const a=s.auction, p=a.to_act, hand=s.players[p].hand, acts=[{kind:'DeclineAdd'}];
    hand.forEach((c,i)=>{ if(c.artist===a.artist && c.auction!=='double') acts.push(c.auction==='fixed'?{kind:'AddSecond',card_index:i,price:null}:{kind:'AddSecond',card_index:i}); });
    return acts;
  }
  if (ph==='bid_open'||ph==='bid_once') {
    const a=s.auction, p=a.to_act, acts=[{kind:'PassBid'}];
    if (s.players[p].money >= a.high+1) acts.push({kind:'Bid',amount:null});
    return acts;
  }
  if (ph==='bid_sealed') return [{kind:'SealedBid',amount:null}];
  if (ph==='buy_fixed') { const a=s.auction, p=a.to_act, acts=[{kind:'PassBuy'}]; if(s.players[p].money>=a.high) acts.push({kind:'Buy'}); return acts; }
  return [];
}

// ── apply ───────────────────────────────────────────────────────────
function apply(state, action) {
  const s = clone(state);
  const ph = s.phase;
  if (ph==='game_over') throw new Error('游戏已结束');
  if (ph==='choose') applyChoose(s, action);
  else if (ph==='double_offer') applyDoubleOffer(s, action);
  else if (ph==='bid_open'||ph==='bid_once') applyBid(s, action);
  else if (ph==='bid_sealed') applySealed(s, action);
  else if (ph==='buy_fixed') applyBuyFixed(s, action);
  else throw new Error('未知阶段 '+ph);
  return s;
}

function placeCard(s, artist) { s.round_counts[artist]+=1; return s.round_counts[artist] >= TRIGGER; }

function validatePrice(s, seller, price) {
  if (price==null) throw new Error('一口价必须定价');
  if (price<0) throw new Error('定价不能为负');
  if (price > s.players[seller].money) throw new Error('定价超过卖家现金');
}

function applyChoose(s, action) {
  if (action.kind!=='ChooseCard') throw new Error('CHOOSE 只接受 ChooseCard');
  const p=s.active_player, hand=s.players[p].hand, i=action.card_index;
  if (!(i>=0 && i<hand.length)) throw new Error('手牌索引越界');
  const card=hand[i];
  if (card.auction==='fixed') validatePrice(s, p, action.price);
  hand.splice(i,1);
  if (placeCard(s, card.artist)) { s.log.push(`P${p} 打出第5张【${card.artist}】→ 回合结束,不拍卖`); endRound(s, p); return; }
  if (card.auction==='double') startDoubleOffer(s, card, p);
  else startAuction(s, card.artist, card.auction, [card], p, p, action.price);
}

function startAuction(s, artist, atype, cards, turnHolder, seller, price) {
  const n=s.num_players;
  const a={artist, auction_type:atype, cards, turn_holder:turnHolder, seller, order:[], to_act:null, idx:0,
           high:0, high_bidder:null, active:[], pass_streak:0, sealed_bids:{}, player_bids:{}, double_card:null, offer_order:[]};
  if (atype==='open') { a.order=rotation(n,(seller+1)%n); a.active=Array.from({length:n},(_,i)=>i); a.to_act=a.order[0]; s.phase='bid_open'; }
  else if (atype==='once') { a.order=rotation(n,(seller+1)%n); a.to_act=a.order[0]; s.phase='bid_once'; }
  else if (atype==='sealed') { a.order=rotation(n,seller); a.to_act=a.order[0]; s.phase='bid_sealed'; }
  else if (atype==='fixed') { a.high=price; a.order=Array.from({length:n-1},(_,k)=>(seller+1+k)%n); a.to_act=a.order[0]; s.phase='buy_fixed'; }
  else throw new Error('不能直接开始 '+atype);
  s.auction=a; s.log.push(`P${seller} 拍卖【${artist}】×${cards.length}(${atype})`);
}

function startDoubleOffer(s, doubleCard, turnHolder) {
  const n=s.num_players;
  s.auction={artist:doubleCard.artist, auction_type:'double', cards:[], turn_holder:turnHolder, seller:turnHolder,
             order:[], to_act:null, idx:0, high:0, high_bidder:null, active:[], pass_streak:0, sealed_bids:{}, player_bids:{},
             double_card:doubleCard, offer_order:rotation(n,turnHolder)};
  s.auction.to_act=s.auction.offer_order[0]; s.phase='double_offer';
  s.log.push(`P${turnHolder} 打出双张【${doubleCard.artist}】,询问补第二张…`);
}

function applyDoubleOffer(s, action) {
  const a=s.auction, p=a.to_act;
  if (action.kind==='DeclineAdd') {
    a.idx+=1;
    if (a.idx<a.offer_order.length) { a.to_act=a.offer_order[a.idx]; return; }
    s.log.push(`无人为双张【${a.artist}】补牌`); finishNoSale(s); return;
  }
  if (action.kind==='AddSecond') {
    const hand=s.players[p].hand, i=action.card_index;
    if (!(i>=0 && i<hand.length)) throw new Error('索引越界');
    const second=hand[i];
    if (second.artist!==a.artist) throw new Error('补牌须同艺术家');
    if (second.auction==='double') throw new Error('不能用双张作配对');
    if (second.auction==='fixed') validatePrice(s, p, action.price);
    hand.splice(i,1);
    const trig=placeCard(s, a.artist);
    const cards=[a.double_card, second];
    if (trig) { s.log.push(`P${p} 补牌达第5张→回合结束,双张不成交`); endRound(s, p); return; }
    s.log.push(`P${p} 补上第二张【${a.artist}】,成为卖家(所得对分)`);
    startAuction(s, a.artist, second.auction, cards, a.turn_holder, p, action.price);
    return;
  }
  throw new Error('DOUBLE_OFFER 只接受 AddSecond/DeclineAdd');
}

function checkBidAmount(s, p, amount, minAmount) {
  if (amount==null) throw new Error('出价须为具体金额');
  if (amount<minAmount) throw new Error(`出价须≥${minAmount}`);
  if (amount>s.players[p].money) throw new Error('出价超过现金');
  return amount;
}

function applyBid(s, action) {
  const a=s.auction, p=a.to_act;
  if (s.phase==='bid_open') {
    if (action.kind==='Bid') { const amt=checkBidAmount(s,p,action.amount,a.high+1); a.high=amt; a.high_bidder=p; a.player_bids[p]=amt; a.pass_streak=0; s.log.push(`P${p} 出价 ${amt}`); }
    else if (action.kind==='PassBid') { a.pass_streak+=1; s.log.push(`P${p} 过(暂不加价)`); }
    else throw new Error('只接受 Bid/PassBid');
    advanceOpen(s);
  } else { // bid_once
    if (action.kind==='Bid') { const amt=checkBidAmount(s,p,action.amount,a.high+1); a.high=amt; a.high_bidder=p; a.player_bids[p]=amt; s.log.push(`P${p} 出价 ${amt}`); }
    else if (action.kind==='PassBid') { s.log.push(`P${p} 放弃`); }
    else throw new Error('只接受 Bid/PassBid');
    a.idx+=1;
    if (a.idx<a.order.length) a.to_act=a.order[a.idx];
    else resolveHighBid(s);
  }
}

function advanceOpen(s) {
  const a=s.auction, n=s.num_players;
  const eligible = n - (a.high_bidder!==null ? 1 : 0);
  if (a.pass_streak >= eligible) { resolveHighBid(s); return; }
  const nOrder=a.order.length, start=a.order.indexOf(a.to_act);
  for (let step=1; step<=nOrder; step++) { const q=a.order[(start+step)%nOrder]; if (q!==a.high_bidder) { a.to_act=q; return; } }
  resolveHighBid(s);
}

function resolveHighBid(s) {
  const a=s.auction;
  if (a.high_bidder!==null) award(s, a.high_bidder, a.high);
  else award(s, a.seller, 0);
}

function applySealed(s, action) {
  const a=s.auction, p=a.to_act;
  if (action.kind!=='SealedBid') throw new Error('暗标只接受 SealedBid');
  const amt=action.amount;
  if (amt==null) throw new Error('暗标须给金额');
  if (amt<0) throw new Error('暗标不能为负');
  if (amt>s.players[p].money) throw new Error('暗标超过现金');
  a.sealed_bids[p]=amt; a.idx+=1;
  if (a.idx<a.order.length) { a.to_act=a.order[a.idx]; return; }
  const best=Math.max(...a.order.map(q=>a.sealed_bids[q]));
  const win=a.order.find(q=>a.sealed_bids[q]===best);
  s.log.push('暗标开标 ['+a.order.map(q=>`P${q}:${a.sealed_bids[q]}`).join(', ')+']');
  award(s, win, best);
}

function applyBuyFixed(s, action) {
  const a=s.auction, p=a.to_act, price=a.high;
  if (action.kind==='Buy') { if (s.players[p].money<price) throw new Error('现金不足'); award(s, p, price); return; }
  if (action.kind==='PassBuy') {
    a.idx+=1;
    if (a.idx<a.order.length) { a.to_act=a.order[a.idx]; return; }
    s.log.push(`无人按 ${price} 购买→卖家 P${a.seller} 自购`); award(s, a.seller, price); return;
  }
  throw new Error('一口价只接受 Buy/PassBuy');
}

function award(s, win, price) {
  const a=s.auction, seller=a.seller, th=a.turn_holder;
  s.players[win].money -= price; s.players[win].paid_total += price;
  if (seller===th) {
    if (win!==seller) s.players[seller].money += price;
    s.log.push(`→ P${win} 以 ${price} 得【${a.artist}】×${a.cards.length}`);
  } else {
    const h=Math.floor(price/2), r=price-h;
    if (win===th) s.players[seller].money += r;
    else if (win===seller) s.players[th].money += h;
    else { s.players[th].money += h; s.players[seller].money += r; }
    s.log.push(`→ P${win} 以 ${price} 得【${a.artist}】×${a.cards.length}(对分 P${th}+${h}/P${seller}+${r})`);
  }
  for (const c of a.cards) s.players[win].purchases.push(c);
  s.auction=null; afterAuction(s, seller);
}

function finishNoSale(s) {
  const a=s.auction, th=a.turn_holder;
  if (a.double_card!=null) { s.players[th].purchases.push(a.double_card); s.log.push(`→ P${th} 免费获得【${a.artist}】`); }
  s.auction=null; afterAuction(s, th);
}

function firstWithCardsFrom(s, start) {
  const n=s.num_players;
  for (let k=0;k<n;k++){ const p=(start+k)%n; if (s.players[p].hand.length) return p; }
  return null;
}
function afterAuction(s, fromPlayer) {
  const nxt=firstWithCardsFrom(s, (fromPlayer+1)%s.num_players);
  if (nxt===null) { endRound(s, fromPlayer); return; }
  s.active_player=nxt; s.phase='choose';
}

function endRound(s, lastPlayer) {
  s.auction=null;
  const counts=s.round_counts;
  const ranked=ARTISTS.filter(a=>counts[a]>0).sort((x,y)=> (counts[y]-counts[x]) || (AIDX[x]-AIDX[y]));
  const top3=ranked.slice(0,3);
  top3.forEach((artist,rank)=>{ s.value_board[artist]+=RANK_VALUES[rank]; s.value_markers[artist][s.round-1]=RANK_VALUES[rank]; });
  s.log.push(`第 ${s.round} 回合结算:前三 ${top3.join('/')||'(无)'}`);
  const top3set=new Set(top3);
  for (let p=0;p<s.num_players;p++){
    let gained=0;
    for (const c of s.players[p].purchases) if (top3set.has(c.artist)) gained += s.value_board[c.artist];
    if (gained){ s.players[p].money+=gained; s.log.push(`  P${p} 变现 +${gained}`); }
    s.players[p].purchases=[];
  }
  for (const a of ARTISTS) s.round_counts[a]=0;
  if (s.round>=NUM_ROUNDS) { s.phase='game_over'; s.log.push(`游戏结束。现金:${finalScores(s)}`); return; }
  s.round+=1; s.start_player=(lastPlayer+1)%s.num_players; deal(s, s.round);
  s.log.push(`—— 第 ${s.round} 回合开始 ——`); openRound(s);
}
function openRound(s) {
  const first=firstWithCardsFrom(s, s.start_player);
  if (first===null) { endRound(s, (s.start_player-1+s.num_players)%s.num_players); return; }
  s.active_player=first; s.phase='choose';
}

// ── 观测(供渲染)────────────────────────────────────────────────────
function observation(s, me) {
  const a=s.auction;
  let sealedPub=null;
  if (s.phase==='bid_sealed' && a) sealedPub={submitted:Object.keys(a.sealed_bids).map(Number).sort((x,y)=>x-y), my_bid:a.sealed_bids[me]??null};
  return {
    you: me, num_players:s.num_players, round:s.round, phase:s.phase, to_act:currentPlayer(s),
    your_hand: s.players[me].hand.map(c=>({...c})), your_money:s.players[me].money,
    your_purchases: s.players[me].purchases.map(c=>({...c})),
    value_board:{...s.value_board}, value_markers:Object.fromEntries(ARTISTS.map(x=>[x,[...s.value_markers[x]]])),
    round_counts:{...s.round_counts},
    players_public: s.players.map((pp,q)=>({
      hand_size:pp.hand.length, purchases:pp.purchases.map(c=>({...c})), purchases_count:pp.purchases.length,
      paid_total:pp.paid_total, money: q===me ? pp.money : null,
    })),
    auction: a ? auctionPublic(s, a) : null, sealed: sealedPub,
    is_over: isOver(s), scores: isOver(s)?finalScores(s):null,
  };
}
function auctionPublic(s, a) {
  const d={artist:a.artist, auction_type:a.auction_type, cards:a.cards.map(c=>({...c})),
    num_cards: a.cards.length || (a.double_card?2:1), is_double_lot: a.cards.length===2,
    seller:a.seller, turn_holder:a.turn_holder, to_act:a.to_act, high:a.high, high_bidder:a.high_bidder};
  if (s.phase==='double_offer') d.double_card = a.double_card?{...a.double_card}:null;
  return d;
}

// ── 观测编码(对齐 rl/encoding.py, OBS_DIM=161)──────────────────────
function projectedValue(s, artist) {
  const counts=s.round_counts;
  const ranked=[...ARTISTS].sort((x,y)=> (counts[y]-counts[x]) || (AIDX[x]-AIDX[y]));
  const pos=ranked.indexOf(artist);
  const bonus = pos<3 ? RANK_VALUES[pos] : 5;
  return s.value_board[artist]+bonus;
}
const maxRoundValue = (s, artist) => s.value_board[artist] + RANK_VALUES[0];
function valueCap(s, me, artist, ncards) { return Math.max(0, Math.min(s.players[me].money, ncards*maxRoundValue(s, artist))); }

// 离散动作布局
const CHOOSE_BASE=0, ADD_BASE=NA*NT, DECLINE_ADD=ADD_BASE+NT, BID=DECLINE_ADD+1,
      PASS_BID=BID+1, SEALED_BID=PASS_BID+1, BUY=SEALED_BID+1, PASS_BUY=BUY+1, NUM_DISCRETE=PASS_BUY+1;

function encodeObs(s, me) {
  const n=s.num_players, rel=Array.from({length:n},(_,k)=>(me+k)%n), f=[];
  // 1) 手牌 (25)+(5)
  const hct={}; for (const c of s.players[me].hand) { const k=c.artist+'|'+c.auction; hct[k]=(hct[k]||0)+1; }
  for (const a of ARTISTS) for (const t of TYPES) f.push((hct[a+'|'+t]||0)/4);
  for (const a of ARTISTS){ let sum=0; for (const t of TYPES) sum+=(hct[a+'|'+t]||0); f.push(sum/8); }
  // 2) 现金
  f.push(s.players[me].money/200);
  // 3) 回合(4)/阶段(7)/人数(3)
  for (let r=1;r<=4;r++) f.push(s.round===r?1:0);
  const PH=['choose','double_offer','bid_open','bid_once','bid_sealed','buy_fixed','game_over'];
  for (const ph of PH) f.push(s.phase===ph?1:0);
  for (const pc of [3,4,5]) f.push(n===pc?1:0);
  // 4) 价值板
  for (const a of ARTISTS) for (let r=0;r<4;r++) f.push(s.value_markers[a][r]/30);
  for (const a of ARTISTS) f.push(s.value_board[a]/90);
  for (const a of ARTISTS) f.push(s.round_counts[a]/5);
  // 5) 各座位(MAXP×9)
  for (let slot=0;slot<MAXP;slot++){
    if (slot<n){ const q=rel[slot]; f.push(1); f.push(s.players[q].hand.length/16);
      const col={}; for (const c of s.players[q].purchases) col[c.artist]=(col[c.artist]||0)+1;
      for (const a of ARTISTS) f.push((col[a]||0)/5);
      f.push(s.players[q].purchases.length/8); f.push(s.players[q].paid_total/200);
    } else for (let k=0;k<9;k++) f.push(0);
  }
  // 6) 拍卖上下文
  const a=s.auction;
  if (a){
    f.push(1);
    for (const art of ARTISTS) f.push(a.artist===art?1:0);
    for (const t of TYPES) f.push(a.auction_type===t?1:0);
    f.push(a.cards.length===2?1:0); f.push((a.cards.length||1)/2);
    const srel=rel.indexOf(a.seller); for (let slot=0;slot<MAXP;slot++) f.push(slot===srel?1:0);
    f.push(a.high/200);
    const hb=(a.high_bidder!=null && rel.indexOf(a.high_bidder)>=0)?rel.indexOf(a.high_bidder):-1;
    for (let slot=0;slot<MAXP;slot++) f.push(slot===hb?1:0);
    f.push(a.high_bidder===null?1:0);
    for (const t of TYPES) f.push((a.double_card && a.double_card.auction===t)?1:0);
    f.push(projectedValue(s, a.artist)/90);
    for (let slot=0;slot<MAXP;slot++){ if (slot<n){ const q=rel[slot]; f.push((a.player_bids[q]||0)/200); f.push(a.active.includes(q)?1:0);} else { f.push(0); f.push(0);} }
  } else {
    const zlen=1+NA+NT+1+1+MAXP+1+MAXP+1+NT+1+MAXP*2;
    for (let k=0;k<zlen;k++) f.push(0);
  }
  return Float32Array.from(f);
}

function legalMask(s, me) {
  const mask=new Array(NUM_DISCRETE).fill(false), ph=s.phase, hand=s.players[me].hand;
  if (ph==='choose') { const seen=new Set(); for (const c of hand) seen.add(c.artist+'|'+c.auction); for (const k of seen){ const [art,t]=k.split('|'); mask[CHOOSE_BASE+AIDX[art]*NT+TIDX[t]]=true; } }
  else if (ph==='double_offer') { mask[DECLINE_ADD]=true; const art=s.auction.artist; for (const c of hand) if (c.artist===art && c.auction!=='double') mask[ADD_BASE+TIDX[c.auction]]=true; }
  else if (ph==='bid_open'||ph==='bid_once') { mask[PASS_BID]=true; const [lo,hi]=bidBounds(s); if (hi>=lo) mask[BID]=true; }
  else if (ph==='bid_sealed') mask[SEALED_BID]=true;
  else if (ph==='buy_fixed') { mask[PASS_BUY]=true; if (s.players[me].money>=s.auction.high) mask[BUY]=true; }
  return mask;
}

function amountRange(s, me, d) {
  if (d>=CHOOSE_BASE && d<ADD_BASE) { const t=TYPES[(d-CHOOSE_BASE)%NT]; if (t!=='fixed') return null; const art=ARTISTS[Math.floor((d-CHOOSE_BASE)/NT)]; return [0, valueCap(s,me,art,1)]; }
  if (d>=ADD_BASE && d<DECLINE_ADD) { const t=TYPES[d-ADD_BASE]; if (t!=='fixed') return null; return [0, valueCap(s,me,s.auction.artist,2)]; }
  if (d===BID) { const a=s.auction, lo=a.high+1; return [lo, Math.max(lo, valueCap(s,me,a.artist,Math.max(1,a.cards.length)))]; }
  if (d===SEALED_BID) { const a=s.auction; return [0, valueCap(s,me,a.artist,Math.max(1,a.cards.length))]; }
  return null;
}
const needsAmount = (s, me, d) => amountRange(s, me, d)!==null;
// 复刻 Python round():逢 .5 取偶(banker's rounding),避免与 Python 逐 1 之差
function pyRound(x) { const f=Math.floor(x), diff=x-f; if (diff<0.5) return f; if (diff>0.5) return f+1; return (f%2===0)?f:f+1; }
function amountFrom01(rng, x) { const [lo,hi]=rng; if (hi<lo) return lo; x=Math.min(1,Math.max(0,x)); return pyRound(lo + x*(hi-lo)); }

function decodeAction(s, me, d, amount01) {
  const hand=s.players[me].hand, rng=amountRange(s,me,d), amt=(rng!=null)?amountFrom01(rng, amount01||0):null;
  const findHand=(art,t)=>{ for (let i=0;i<hand.length;i++) if (hand[i].artist===art && hand[i].auction===t) return i; throw new Error('手牌找不到 '+art+'/'+t); };
  if (d>=CHOOSE_BASE && d<ADD_BASE) { const art=ARTISTS[Math.floor((d-CHOOSE_BASE)/NT)], t=TYPES[(d-CHOOSE_BASE)%NT]; const i=findHand(art,t); return t==='fixed'?{kind:'ChooseCard',card_index:i,price:amt}:{kind:'ChooseCard',card_index:i}; }
  if (d>=ADD_BASE && d<DECLINE_ADD) { const t=TYPES[d-ADD_BASE], art=s.auction.artist, i=findHand(art,t); return t==='fixed'?{kind:'AddSecond',card_index:i,price:amt}:{kind:'AddSecond',card_index:i}; }
  if (d===DECLINE_ADD) return {kind:'DeclineAdd'};
  if (d===BID) return {kind:'Bid',amount:amt};
  if (d===PASS_BID) return {kind:'PassBid'};
  if (d===SEALED_BID) return {kind:'SealedBid',amount:amt};
  if (d===BUY) return {kind:'Buy'};
  if (d===PASS_BUY) return {kind:'PassBuy'};
  throw new Error('未知离散动作 '+d);
}

const API = {
  ARTISTS, TYPES, NUM_DISCRETE, clone, newGame, isOver, currentPlayer, finalScores, winner, bidBounds,
  legalActions, apply, observation, encodeObs, legalMask, amountRange, needsAmount, decodeAction,
  projectedValue, maxRoundValue,
};
if (typeof module!=='undefined' && module.exports) module.exports = API;
root.MAEngine = API;
})(typeof window!=='undefined' ? window : globalThis);
