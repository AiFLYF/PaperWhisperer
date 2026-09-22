/* ============================================================
   PaperWhisperer — 纸上之墨
   滚动显现 / 页码跟随 / 阅读进度 / 首叶分析演示 / 命令复制
   ============================================================ */

'use strict';

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

if (window.lucide) window.lucide.createIcons();

/* ---------- 入场编排 ---------- */
/* 等一帧再掀标题，保证过渡有起点；后台标签页里 rAF 会暂停，用定时器兜底 */
const go = () => document.body.classList.add('is-ready');
requestAnimationFrame(() => requestAnimationFrame(go));
setTimeout(go, 1200);

/* ---------- 滚动显现 ---------- */

const io = new IntersectionObserver((entries) => {
  for (const en of entries) {
    if (en.isIntersecting) { en.target.classList.add('in-view'); io.unobserve(en.target); }
  }
}, { threshold: 0.15, rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('[data-reveal]').forEach((el) => io.observe(el));

/* ---------- 页码 / 导航跟随 + 阅读进度 ---------- */

const sections = [...document.querySelectorAll('[data-sec]')];
const folioLinks = [...document.querySelectorAll('.folio-rail a')];
const navLinks = [...document.querySelectorAll('.site-nav a[data-target]')];
const readBar = document.getElementById('readBar');
const thread = document.getElementById('thread');
const threadFill = document.getElementById('threadFill');

/* 红绳上的章节刻度：按各章在纸面上的位置落点，一次量好 */
if (thread) {
  const H = thread.offsetHeight || 1;
  sections.forEach((s) => {
    const t = document.createElement('b');
    t.style.top = `${Math.min(100, (s.offsetTop / H) * 100)}%`;
    thread.appendChild(t);
  });
}

function setActive(i) {
  folioLinks.forEach((a, k) => a.classList.toggle('active', k === i));
  // 顶部导航没有"首叶"项：data-target 1..3 对应 sections 1..3
  navLinks.forEach((a, k) => a.classList.toggle('active', k + 1 === i));
}

let ticking = false;
function onScroll() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => {
    ticking = false;
    const doc = document.documentElement;
    const max = doc.scrollHeight - innerHeight;
    const p = max > 0 ? scrollY / max : 0;
    if (readBar) readBar.style.transform = `scaleX(${p})`;
    if (threadFill) threadFill.style.transform = `scaleY(${p})`;
    const probe = scrollY + innerHeight * 0.35;
    let i = 0;
    for (let k = 0; k < sections.length; k++) {
      if (sections[k].offsetTop <= probe) i = k;
    }
    setActive(i);
  });
}
addEventListener('scroll', onScroll, { passive: true });
onScroll();

/* 锚点跳转（原生 scroll-behavior 已平滑，这里只处理 reduced-motion 下的一致性与 sticky 头部偏移） */
document.querySelectorAll('[data-target]').forEach((el) => {
  el.addEventListener('click', (e) => {
    const i = parseInt(el.dataset.target, 10);
    const sec = sections[i];
    if (!sec) return;
    e.preventDefault();
    const y = sec.offsetTop - 46;
    scrollTo({ top: Math.max(0, y), behavior: reduced ? 'auto' : 'smooth' });
    history.replaceState(null, '', sec.id ? '#' + sec.id : ' ');
  });
});

/* ---------- 首叶：分析演示 ---------- */

const reader = document.getElementById('reader');

const DEMO = {
  status: [
    '等待上传…',
    '已接收 1706.03762v7.pdf · 校验通过',
    '抽取正文 · 15 页 / 4.2 MB',
    '六段任务并发中…',
    '结论已齐 · 可以继续追问',
  ],
  // 各行的出场顺序与打完后停顿（毫秒）：并发的感觉来自"不等前面的打完"
  rows: [
    { idx: 1, start: 2600, speed: 26 },  // 概览
    { idx: 3, start: 3050, speed: 22 },  // 文本结构
    { idx: 2, start: 3600, speed: 30 },  // 关键引用
    { idx: 5, start: 4050, speed: 24 },  // 批判评价
    { idx: 0, start: 4700, speed: 21 },  // 视觉图谱
    { idx: 4, start: 5300, speed: 27 },  // 深度简报
  ],
};

function runDemo() {
  const el = reader;
  if (!el) return;
  const statusEl = document.getElementById('rdStatus');
  const elapsedEl = document.getElementById('rdElapsed');
  const stampEl = document.getElementById('rdStamp');
  const replayBtn = document.getElementById('rdReplay');
  const rowEls = [...el.querySelectorAll('.rd-row')];
  // 重播时清掉上一场还没烧完的定时器
  rowEls.forEach((r) => clearTimeout(r._t));
  if (runDemo._timers) runDemo._timers.forEach((t) => clearTimeout(t));
  const timers = [];
  runDemo._timers = timers;
  const at = (ms, fn) => timers.push(setTimeout(fn, ms));

  // 复位
  rowEls.forEach((r) => {
    r.classList.remove('lit', 'typing', 'done');
    r.querySelector('.rd-v').textContent = '';
  });
  stampEl.classList.remove('stamped');
  replayBtn.hidden = true;
  elapsedEl.textContent = '00:00';
  statusEl.textContent = DEMO.status[0];

  if (reduced) {
    // 减少动态：直接展示最终态
    rowEls.forEach((r) => {
      r.querySelector('.rd-v').textContent = r.querySelector('.rd-v').dataset.full;
      r.classList.add('lit', 'done');
    });
    statusEl.textContent = DEMO.status[4];
    stampEl.classList.add('stamped');
    return;
  }

  // 计时器：01:58 → 03:42（模拟真实耗时）
  const t0 = performance.now();
  const tickTimer = setInterval(() => {
    const p = Math.min(1, (performance.now() - t0) / 8600);
    const secs = Math.round(118 + p * 104);
    elapsedEl.textContent =
      String(Math.floor(secs / 60)).padStart(2, '0') + ':' + String(secs % 60).padStart(2, '0');
  }, 250);
  at(8600, () => clearInterval(tickTimer));
  timers.push(tickTimer);

  at(700, () => (statusEl.textContent = DEMO.status[1]));
  at(1600, () => (statusEl.textContent = DEMO.status[2]));
  at(2500, () => (statusEl.textContent = DEMO.status[3]));

  // 逐字符打字机
  function type(row) {
    const vEl = row.querySelector('.rd-v');
    const text = vEl.dataset.full;
    row.classList.add('lit', 'typing');
    let i = 0;
    const step = () => {
      i++;
      vEl.textContent = text.slice(0, i);
      if (i < text.length) {
        row._t = setTimeout(step, row._speed * (0.65 + Math.random() * 0.7));
        timers.push(row._t);
      } else {
        row.classList.remove('typing');
        row.classList.add('done');
      }
    };
    row._t = setTimeout(step, row._speed);
    timers.push(row._t);
  }

  for (const r of DEMO.rows) {
    const row = rowEls[r.idx];
    row._speed = r.speed;
    at(r.start, () => type(row));
  }

  at(7600, () => (statusEl.textContent = DEMO.status[4]));
  at(8000, () => {
    stampEl.classList.add('stamped');
    replayBtn.hidden = false;
  });
}

// 只播一次进入视口的那场；之后由"重新演示"接管
let demoPlayed = false;
const demoIo = new IntersectionObserver((entries) => {
  for (const en of entries) {
    if (en.isIntersecting && !demoPlayed) {
      demoPlayed = true;
      runDemo();
      demoIo.disconnect();
    }
  }
}, { threshold: 0.35 });
if (reader) demoIo.observe(reader);

const replayBtn = document.getElementById('rdReplay');
if (replayBtn) replayBtn.addEventListener('click', runDemo);

/* ---------- 命令块复制 ---------- */

const copyBtn = document.getElementById('copyBtn');
if (copyBtn) {
  copyBtn.addEventListener('click', async () => {
    const cmd = [
      'pip install -r requirements.txt',
      'cp .env.example .env',
      'python web_app.py',
    ].join('\n');
    try {
      await navigator.clipboard.writeText(cmd);
      copyBtn.classList.add('copied');
      copyBtn.innerHTML = '<i data-lucide="check"></i> 已复制';
      if (window.lucide) window.lucide.createIcons();
      setTimeout(() => {
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = '<i data-lucide="clipboard-copy"></i> 复制';
        if (window.lucide) window.lucide.createIcons();
      }, 2200);
    } catch {
      /* 剪贴板不可用时保持原文，点击即选中代码块也无碍 */
    }
  });
}
