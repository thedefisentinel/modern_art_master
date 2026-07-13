/* JS 版智能体(对齐 modern_art/agents)。基线为纯逻辑;rl 走 ONNX。
   每个 agent 暴露 async act(state) -> 动作对象(可直接给 MAEngine.apply)。 */
(function (root) {
'use strict';
const E = root.MAEngine || (typeof require !== 'undefined' && require('./engine.js'));
const RANK0 = 30;
const proj = (s, a) => E.projectedValue(s, a);
const maxVal = (s, a) => E.maxRoundValue(s, a);
const TYPE_PREF = ['once', 'open', 'sealed', 'fixed', 'double'];
const randint = (rnd, lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));
function mulberry32(seed){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

// ── 随机 ──────────────────────────────────────────────────────────
function randomAgent(rnd) {
  return { act: (s) => {
    const me = E.currentPlayer(s), acts = E.legalActions(s), a = acts[Math.floor(rnd() * acts.length)];
    const money = s.players[me].money;
    if (a.kind === 'ChooseCard') { const c = s.players[me].hand[a.card_index]; return c.auction === 'fixed' ? {kind:'ChooseCard', card_index:a.card_index, price:randint(rnd,0,money)} : {kind:'ChooseCard', card_index:a.card_index}; }
    if (a.kind === 'AddSecond') { const c = s.players[me].hand[a.card_index]; return c.auction === 'fixed' ? {kind:'AddSecond', card_index:a.card_index, price:randint(rnd,0,money)} : {kind:'AddSecond', card_index:a.card_index}; }
    if (a.kind === 'Bid') { const [lo,hi]=E.bidBounds(s); return {kind:'Bid', amount:randint(rnd, lo, Math.min(hi, lo+40))}; }
    if (a.kind === 'SealedBid') { const [lo,hi]=E.bidBounds(s); return {kind:'SealedBid', amount:randint(rnd, lo, Math.min(hi,40))}; }
    return a;
  }};
}

// ── 标准(一般) ────────────────────────────────────────────────────
function standardAgent() {
  const willing = (s,art,nc) => proj(s,art) * nc;
  return { act: (s) => {
    const me = E.currentPlayer(s), ph = s.phase, hand = s.players[me].hand, a = s.auction;
    if (ph === 'choose') {
      const cnt = {}; hand.forEach(c => cnt[c.artist] = (cnt[c.artist]||0)+1);
      let bi=0, bk=[-1,-1,99];
      hand.forEach((c,i) => { const key=[proj(s,c.artist), cnt[c.artist], -TYPE_PREF.indexOf(c.auction)]; if (key[0]>bk[0]||(key[0]===bk[0]&&(key[1]>bk[1]||(key[1]===bk[1]&&key[2]>bk[2])))) { bk=key; bi=i; } });
      const c=hand[bi];
      if (c.auction==='fixed') return {kind:'ChooseCard',card_index:bi,price:Math.max(1,Math.min(s.players[me].money,Math.floor(proj(s,c.artist)*0.8)))};
      return {kind:'ChooseCard',card_index:bi};
    }
    if (ph === 'double_offer') {
      const adds = E.legalActions(s).filter(x=>x.kind==='AddSecond');
      if (!adds.length) return {kind:'DeclineAdd'};
      adds.sort((x,y)=>TYPE_PREF.indexOf(hand[x.card_index].auction)-TYPE_PREF.indexOf(hand[y.card_index].auction));
      const add=adds[0], c=hand[add.card_index];
      if (c.auction==='fixed') return {kind:'AddSecond',card_index:add.card_index,price:Math.max(1,Math.min(s.players[me].money,Math.floor(proj(s,c.artist)*0.8)))};
      return {kind:'AddSecond',card_index:add.card_index};
    }
    if (ph==='bid_open'||ph==='bid_once') { const [lo,hi]=E.bidBounds(s), w=willing(s,a.artist,a.cards.length); if (me===a.seller) return {kind:'PassBid'}; return (lo<=w&&lo<=hi)?{kind:'Bid',amount:lo}:{kind:'PassBid'}; }
    if (ph==='bid_sealed') { const [lo,hi]=E.bidBounds(s), w=willing(s,a.artist,a.cards.length); return {kind:'SealedBid',amount:Math.max(0,Math.min(hi,Math.floor(w*0.6)))}; }
    if (ph==='buy_fixed') { const w=willing(s,a.artist,a.cards.length); return (a.high<=w&&a.high<=s.players[me].money)?{kind:'Buy'}:{kind:'PassBuy'}; }
    return E.legalActions(s)[0];
  }};
}

// ── 启发式 ──────────────────────────────────────────────────────────
function heuristicAgent() {
  const est = (s,a) => { const c=s.round_counts[a]; const m = c>=4?30:c>=2?20:c>=1?10:0; return s.value_board[a]+m; };
  return { act: (s) => {
    const me=E.currentPlayer(s), ph=s.phase, hand=s.players[me].hand, a=s.auction;
    if (ph==='choose') {
      const cnt={}; hand.forEach(c=>cnt[c.artist]=(cnt[c.artist]||0)+1);
      let bi=0,bk=[-1,-1,99];
      hand.forEach((c,i)=>{const key=[est(s,c.artist),cnt[c.artist],-TYPE_PREF.indexOf(c.auction)]; if(key[0]>bk[0]||(key[0]===bk[0]&&(key[1]>bk[1]||(key[1]===bk[1]&&key[2]>bk[2])))){bk=key;bi=i;}});
      const c=hand[bi];
      if (c.auction==='fixed') return {kind:'ChooseCard',card_index:bi,price:Math.max(0,Math.min(s.players[me].money,Math.floor(est(s,c.artist)*0.6)))};
      return {kind:'ChooseCard',card_index:bi};
    }
    if (ph==='double_offer') { const adds=E.legalActions(s).filter(x=>x.kind==='AddSecond'); if(!adds.length) return {kind:'DeclineAdd'}; if (est(s,a.artist)<12) return {kind:'DeclineAdd'}; adds.sort((x,y)=>TYPE_PREF.indexOf(hand[x.card_index].auction)-TYPE_PREF.indexOf(hand[y.card_index].auction)); const add=adds[0],c=hand[add.card_index]; if(c.auction==='fixed') return {kind:'AddSecond',card_index:add.card_index,price:Math.max(0,Math.min(s.players[me].money,Math.floor(est(s,c.artist)*0.6)))}; return {kind:'AddSecond',card_index:add.card_index}; }
    if (ph==='bid_open'||ph==='bid_once') { const [lo,hi]=E.bidBounds(s), w=est(s,a.artist)*a.cards.length; if(me===a.seller) return {kind:'PassBid'}; return (lo<=w&&lo<=hi)?{kind:'Bid',amount:lo}:{kind:'PassBid'}; }
    if (ph==='bid_sealed') { const [lo,hi]=E.bidBounds(s), w=est(s,a.artist)*a.cards.length; return {kind:'SealedBid',amount:Math.max(0,Math.min(hi,Math.floor(w*0.5)))}; }
    if (ph==='buy_fixed') { const w=est(s,a.artist)*a.cards.length; return (a.high<=w&&a.high<=s.players[me].money)?{kind:'Buy'}:{kind:'PassBuy'}; }
    return E.legalActions(s)[0];
  }};
}

// ── 莽夫(85% 顶格,绝不超上限)─────────────────────────────────────
function aggressiveAgent(rnd, aggr=0.85) {
  const want=(s,me,art,nc)=>Math.max(0,Math.min(s.players[me].money,Math.floor(aggr*nc*maxVal(s,art))));
  return { act:(s)=>{
    const me=E.currentPlayer(s), ph=s.phase, hand=s.players[me].hand, a=s.auction;
    if (ph==='choose') { const i=Math.floor(rnd()*hand.length),c=hand[i]; if(c.auction==='fixed') return {kind:'ChooseCard',card_index:i,price:Math.min(s.players[me].money,Math.max(1,Math.floor(proj(s,c.artist)/2)))}; return {kind:'ChooseCard',card_index:i}; }
    if (ph==='double_offer') { const adds=E.legalActions(s).filter(x=>x.kind==='AddSecond'); if(!adds.length) return {kind:'DeclineAdd'}; const add=adds[0],c=hand[add.card_index]; if(c.auction==='fixed') return {kind:'AddSecond',card_index:add.card_index,price:Math.min(s.players[me].money,Math.max(1,Math.floor(proj(s,c.artist)/2)))}; return {kind:'AddSecond',card_index:add.card_index}; }
    if (ph==='bid_open'||ph==='bid_once') { const [lo,hi]=E.bidBounds(s), w=want(s,me,a.artist,a.cards.length); return (lo<=Math.min(w,hi))?{kind:'Bid',amount:Math.min(hi,w)}:{kind:'PassBid'}; }
    if (ph==='bid_sealed') { const [lo,hi]=E.bidBounds(s), w=want(s,me,a.artist,a.cards.length); return {kind:'SealedBid',amount:Math.max(0,Math.min(hi,w))}; }
    if (ph==='buy_fixed') return s.players[me].money>=a.high?{kind:'Buy'}:{kind:'PassBuy'};
    return E.legalActions(s)[0];
  }};
}

// ── 铁公鸡 ──────────────────────────────────────────────────────────
function tightAgent(rnd) {
  return { act:(s)=>{
    const me=E.currentPlayer(s), ph=s.phase, hand=s.players[me].hand, a=s.auction;
    if (ph==='choose') { const i=Math.floor(rnd()*hand.length),c=hand[i]; if(c.auction==='fixed') return {kind:'ChooseCard',card_index:i,price:Math.min(s.players[me].money,Math.max(1,proj(s,c.artist)))}; return {kind:'ChooseCard',card_index:i}; }
    if (ph==='double_offer') return {kind:'DeclineAdd'};
    if (ph==='bid_open'||ph==='bid_once') { const [lo,hi]=E.bidBounds(s); return (lo<=hi&&lo<=Math.floor(proj(s,a.artist)*a.cards.length/2))?{kind:'Bid',amount:lo}:{kind:'PassBid'}; }
    if (ph==='bid_sealed') { const [lo]=E.bidBounds(s); return {kind:'SealedBid',amount:lo}; }
    if (ph==='buy_fixed') return (s.players[me].money>=a.high&&a.high<=proj(s,a.artist))?{kind:'Buy'}:{kind:'PassBuy'};
    return E.legalActions(s)[0];
  }};
}

// ── RL「AI 大师」(ONNX 贪心)──────────────────────────────────────
function rlAgent(session) {
  const EPS=1e-4;
  return { act: async (s) => {
    const me=E.currentPlayer(s);
    const obs=E.encodeObs(s, me);
    const mask=E.legalMask(s, me);
    const feeds={obs:new root.ort.Tensor('float32', obs, [1, obs.length])};
    const out=await session.run(feeds);
    const logits=out.logits.data, alpha=out.alpha.data, beta=out.beta.data;
    let best=-Infinity, d=0;
    for (let i=0;i<logits.length;i++){ const v=mask[i]?logits[i]:-Infinity; if (v>best){best=v; d=i;} }
    let mean=alpha[0]/(alpha[0]+beta[0]); mean=Math.min(1-EPS,Math.max(EPS,mean));
    return E.decodeAction(s, me, d, mean);
  }};
}

// makeAgent(type, seed, session)
function makeAgent(type, seed, session) {
  const rnd = mulberry32((seed==null?(Math.random()*1e9|0):seed) >>> 0);
  switch (type) {
    case 'random': return randomAgent(rnd);
    case 'standard': return standardAgent();
    case 'heuristic': return heuristicAgent();
    case 'aggressive': return aggressiveAgent(rnd);
    case 'tight': return tightAgent(rnd);
    case 'rl': return rlAgent(session);
    default: throw new Error('未知 AI 类型 '+type);
  }
}

const API = { makeAgent };
if (typeof module!=='undefined' && module.exports) module.exports = API;
root.MAAgents = API;
})(typeof window!=='undefined' ? window : globalThis);
