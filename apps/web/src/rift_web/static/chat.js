// Chat page: POST /api/chat and read the SSE stream (EventSource cannot POST).
(() => {
  "use strict";

  const KEY = "rift.session";
  const messages = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = form.querySelector("button[type=submit]");
  const newBtn = document.getElementById("new-session");
  const MODE = { ranked_solo_duo: "单双排", ranked_flex: "灵活组排", normal_draft: "匹配",
                 aram: "大乱斗", aram_mayhem: "海克斯大乱斗", arena: "斗魂竞技场" };
  const RANK = { iron: "黑铁", bronze: "青铜", silver: "白银", gold: "黄金", platinum: "铂金",
                 emerald: "翡翠", diamond: "钻石", master: "大师", grandmaster: "宗师", challenger: "王者" };
  const ROLE = { top: "上单", jungle: "打野", mid: "中单", adc: "ADC", support: "辅助" };
  let busy = false;

  function newSessionId() {
    const raw = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now());
    return raw.replace(/[^A-Za-z0-9]/g, "").slice(0, 16);
  }

  function sessionId() {
    let id = null;
    try { id = sessionStorage.getItem(KEY); } catch (e) { /* storage disabled */ }
    if (!id) {
      id = newSessionId();
      try { sessionStorage.setItem(KEY, id); } catch (e) { /* ignore */ }
    }
    return id;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function scroll() { messages.scrollTop = messages.scrollHeight; }

  function bubble(role, text) {
    const node = el("div", "msg " + role, text);
    messages.appendChild(node);
    scroll();
    return node;
  }

  function fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getMonth() + 1}月${d.getDate()}日 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function renderCandidates(list) {
    const wrap = el("div", "cards");
    list.forEach((c, i) => {
      const card = el("div", "card cand");
      card.appendChild(el("div", "cand-name", `${i + 1}. ${c.name}`));
      const roles = (c.roles || []).map((r) => ROLE[r] || r).join("/");
      card.appendChild(el("div", "muted", `${RANK[c.rank] || c.rank}｜${roles}｜评分 ${c.rating}`));
      card.appendChild(el("div", "price", `${c.effective_hourly_price} 元/小时`));
      if (c.tags && c.tags.length) card.appendChild(el("div", "tags", c.tags.join(" · ")));
      const btn = el("button", "primary small", "选这位");
      btn.addEventListener("click", () => send(`选${c.name}`));
      card.appendChild(btn);
      wrap.appendChild(card);
    });
    messages.appendChild(wrap);
    scroll();
  }

  function renderConfirm(c) {
    const card = el("div", "card confirm");
    if (c.kind === "cancel") {
      const o = c.order || {};
      card.appendChild(el("div", "title", `取消订单 #${o.booking_id}`));
      card.appendChild(el("div", "", `${o.companion_name}｜${MODE[o.game_mode] || o.game_mode}｜${fmtTime(o.start_time)}`));
      card.appendChild(el("div", "price", c.paid ? `可退 ${c.refund_amount} 元（${c.refund_label}）` : "未支付，取消无费用"));
    } else {
      card.appendChild(el("div", "title", "确认订单"));
      card.appendChild(el("div", "", `${c.companion}｜${MODE[c.mode] || c.mode}｜${fmtTime(c.start_time)} 起 ${Number(c.hours)} 小时`));
      card.appendChild(el("div", "price", `${c.unit_price} × ${c.multiplier} × ${Number(c.hours)} = ${c.total} 元`));
    }
    const row = el("div", "row");
    const yes = el("button", "primary small", c.kind === "cancel" ? "确认取消" : "确认下单");
    const no = el("button", "ghost small", c.kind === "cancel" ? "保留订单" : "换一位");
    yes.addEventListener("click", () => send("确认"));
    no.addEventListener("click", () => send("算了"));
    row.append(yes, no);
    card.appendChild(row);
    messages.appendChild(card);
    scroll();
  }

  function renderBooked(o) {
    const card = el("div", "card booked");
    card.appendChild(el("div", "title", `下单成功 #${o.booking_id}`));
    const link = el("a", "", "去「我的订单」支付 →");
    link.href = "/bookings";
    card.appendChild(link);
    messages.appendChild(card);
    scroll();
  }

  function handle(event, data, state) {
    if (event === "token") {
      if (!state.bot) { state.typing.remove(); state.bot = bubble("bot", ""); }
      state.bot.textContent += data.text;
      scroll();
    } else if (event === "candidates") {
      renderCandidates(data.candidates || []);
    } else if (event === "confirm") {
      renderConfirm(data);
    } else if (event === "booked") {
      renderBooked(data);
    } else if (event === "error") {
      state.typing.remove();
      bubble("error", data.message || "出错了，请稍后再试");
      state.failed = true;
    }
  }

  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    input.value = "";
    sendBtn.disabled = true;
    bubble("user", text);
    const state = { bot: null, typing: bubble("bot typing", "…"), failed: false };
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ session_id: sessionId(), message: text }),
      });
      if (resp.status === 401) { location.href = "/login?next=/chat"; return; }
      if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, cut);
          buf = buf.slice(cut + 2);
          let event = "message", data = "";
          frame.split("\n").forEach((line) => {
            if (line.startsWith("event: ")) event = line.slice(7);
            else if (line.startsWith("data: ")) data += line.slice(6);
          });
          handle(event, data ? JSON.parse(data) : {}, state);
        }
      }
      if (!state.bot && !state.failed) state.typing.remove();
    } catch (err) {
      state.typing.remove();
      bubble("error", "网络异常，请稍后再试");
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  async function loadHistory() {
    try {
      const resp = await fetch("/api/chat/history?session_id=" + encodeURIComponent(sessionId()), { credentials: "same-origin" });
      if (!resp.ok) return;
      const data = await resp.json();
      (data.messages || []).forEach((m) => bubble(m.role === "user" ? "user" : "bot", m.content));
    } catch (e) { /* history is best effort */ }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  if (newBtn) {
    newBtn.addEventListener("click", () => {
      try { sessionStorage.setItem(KEY, newSessionId()); } catch (e) { /* ignore */ }
      messages.querySelectorAll(".msg:not(.welcome), .cards, .card").forEach((n) => n.remove());
      input.focus();
    });
  }
  window.riftChat = { send, sessionId };
  const prefill = new URLSearchParams(location.search).get("q");
  if (prefill) { input.value = prefill.slice(0, 500); input.focus(); }
  loadHistory();
})();
