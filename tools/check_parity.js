// 用 engine.js 复现 Python 导出的每一步,逐条比对观测/掩码/合法动作/后态。
const fs = require('fs');
const path = require('path');
const E = require(path.join(__dirname, '..', 'modern_art', 'web', 'static', 'engine.js'));

const recs = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity.json'), 'utf8'));

function stripLog(s) { const c = structuredClone(s); delete c.log; return c; }
function deepEq(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function actKey(a) { return JSON.stringify(a, Object.keys(a).sort()); }

let obsFail = 0, maskFail = 0, legalFail = 0, stateFail = 0, maxObsDiff = 0;
for (let i = 0; i < recs.length; i++) {
  const r = recs[i];
  const before = r.before, me = r.me;
  // 观测
  const obs = E.encodeObs(before, me);
  if (obs.length !== r.obs.length) { obsFail++; }
  else for (let k = 0; k < obs.length; k++) { const d = Math.abs(obs[k] - r.obs[k]); if (d > maxObsDiff) maxObsDiff = d; if (d > 1e-4) { obsFail++; break; } }
  // 掩码
  const mask = E.legalMask(before, me);
  if (!deepEq(mask, r.mask)) maskFail++;
  // 合法动作(集合相等)
  const jl = E.legalActions(before).map(actKey).sort();
  const pl = r.legal.map(a => { const c = {}; for (const k of Object.keys(a)) if (a[k] !== null || k === 'kind') c[k] = a[k]; return c; });
  // Python 模板里 price/amount 为 null;JS 模板不含该键或为 null。统一:去掉值为 null 的键再比。
  const norm = (a) => { const c = {}; for (const k of Object.keys(a)) if (a[k] !== null) c[k] = a[k]; return actKey(c); };
  const jlN = E.legalActions(before).map(norm).sort();
  const plN = r.legal.map(norm).sort();
  if (!deepEq(jlN, plN)) { legalFail++; if (legalFail <= 3) console.log('  legal 差异 @', i, '\n   JS:', jlN, '\n   PY:', plN); }
  // 后态
  const after = E.apply(before, r.action);
  if (!deepEq(stripLog(after), stripLog(r.after))) {
    stateFail++;
    if (stateFail <= 3) {
      const A = stripLog(after), B = stripLog(r.after);
      for (const key of Object.keys(B)) if (!deepEq(A[key], B[key])) { console.log('  state 差异 @', i, 'phase', before.phase, 'act', r.action.kind, 'key', key); break; }
    }
  }
}

console.log(`记录 ${recs.length} 条`);
console.log(`观测不符 ${obsFail}  (最大逐位误差 ${maxObsDiff.toExponential(2)})`);
console.log(`掩码不符 ${maskFail}`);
console.log(`合法动作不符 ${legalFail}`);
console.log(`后态不符 ${stateFail}`);
console.log((obsFail + maskFail + legalFail + stateFail === 0) ? 'PARITY OK ✅' : 'PARITY 有差异 ❌');
