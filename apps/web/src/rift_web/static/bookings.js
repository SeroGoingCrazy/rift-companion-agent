// My orders: list, simulated payment, cancellation with a refund preview.
(() => {
  "use strict";

  const list = document.getElementById("bookings");
  const notice = document.getElementById("notice");
  const dialog = document.getElementById("cancel-dialog");
  const dialogText = document.getElementById("cancel-text");
  const MODE = { ranked_solo_duo: "单双排", ranked_flex: "灵活组排", normal_draft: "匹配",
                 aram: "大乱斗", aram_mayhem: "海克斯大乱斗", arena: "斗魂竞技场" };
  const STATUS = { pending_payment: "待支付", confirmed: "已支付", completed: "已完成", cancelled: "已取消" };
  let pendingCancel = null;

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function fmtTime(iso) {
    const d = new Date(iso);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getMonth() + 1}月${d.getDate()}日 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function say(text, isError) {
    notice.textContent = text || "";
    notice.className = "notice" + (isError ? " error" : "");
  }

  async function api(method, url, body) {
    const resp = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (resp.status === 401) { location.href = "/login?next=/bookings"; throw new Error("login"); }
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || "操作失败");
    return data;
  }

  function render(bookings) {
    list.replaceChildren();
    if (!bookings.length) {
      const empty = el("p", "muted", "还没有订单。");
      const link = el("a", "", "去聊天预约一个 →");
      link.href = "/chat";
      empty.appendChild(link);
      list.appendChild(empty);
      return;
    }
    bookings.forEach((b) => {
      const card = el("div", "card order");
      const head = el("div", "order-head");
      head.append(el("span", "title", `#${b.booking_id} ${b.companion_name}`),
                  el("span", "badge " + b.status, STATUS[b.status] || b.status));
      card.appendChild(head);
      card.appendChild(el("div", "", `${MODE[b.game_mode] || b.game_mode}｜${fmtTime(b.start_time)} 起 ${Number(b.hours)} 小时`));
      let money = `${b.total} 元`;
      if (b.status === "cancelled" && b.refund_amount !== null) money += `（已退 ${b.refund_amount} 元）`;
      card.appendChild(el("div", "price", money));
      const row = el("div", "row");
      if (b.status === "pending_payment") {
        const pay = el("button", "primary small", "模拟支付");
        pay.dataset.action = "pay";
        pay.addEventListener("click", () => doPay(b.booking_id, pay));
        row.appendChild(pay);
      }
      if (b.status === "pending_payment" || b.status === "confirmed") {
        const cancel = el("button", "ghost small", "取消订单");
        cancel.dataset.action = "cancel";
        cancel.addEventListener("click", () => askCancel(b.booking_id, cancel));
        row.appendChild(cancel);
      }
      card.appendChild(row);
      list.appendChild(card);
    });
  }

  async function load() {
    list.classList.add("loading");
    try {
      render((await api("GET", "/api/bookings")).bookings);
    } catch (e) {
      if (e.message !== "login") say("订单加载失败：" + e.message, true);
    } finally {
      list.classList.remove("loading");
    }
  }

  async function doPay(id, btn) {
    btn.disabled = true;
    try {
      await api("POST", `/api/bookings/${id}/pay`);
      say(`订单 #${id} 支付成功`);
      await load();
    } catch (e) { say(e.message, true); btn.disabled = false; }
  }

  async function askCancel(id, btn) {
    btn.disabled = true;
    try {
      const q = await api("POST", `/api/bookings/${id}/cancel`, { dry_run: true });
      pendingCancel = id;
      dialogText.textContent = q.paid
        ? `距开局 ${q.hours_before_start} 小时，${q.refund_label}，可退 ${q.refund_amount} 元。确定取消订单 #${id} 吗？`
        : `订单 #${id} 尚未支付，取消不产生费用。确定取消吗？`;
      dialog.showModal ? dialog.showModal() : dialog.setAttribute("open", "");
    } catch (e) { say(e.message, true); }
    btn.disabled = false;
  }

  document.getElementById("cancel-yes").addEventListener("click", async () => {
    const id = pendingCancel;
    dialog.close ? dialog.close() : dialog.removeAttribute("open");
    if (id === null) return;
    try {
      const done = await api("POST", `/api/bookings/${id}/cancel`, { dry_run: false });
      say(`订单 #${id} 已取消` + (done.paid ? `，退款 ${done.refund_amount} 元` : ""));
      await load();
    } catch (e) { say(e.message, true); }
    pendingCancel = null;
  });
  document.getElementById("cancel-no").addEventListener("click", () => {
    pendingCancel = null;
    dialog.close ? dialog.close() : dialog.removeAttribute("open");
  });

  load();
})();
