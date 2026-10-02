/* ============================================================================
 * Governed 控制台前端（零构建、原生 JS、hash 路由）
 *
 * 设计约束（见 docs/控制台与事件协议设计.md）：
 * - 只渲染框架概念（任务 / 子任务 / 工具 / 角色 / 管控层 / 事件 / 产物契约），
 *   不出现任何领域名词；领域贡献走插件页的 contributes。
 * - "事件流 + 状态快照两条腿"：SSE 只表达发生了什么，终态 / 待审批以快照为准。
 * - 状态一律用「符号 + 文字」chip，不只靠颜色。
 * ========================================================================== */
(function () {
  "use strict";

  var API = "/api/v1";

  /* ---------------- 访问令牌 ----------------
   * 控制台外壳是匿名的，但数据一律走受保护的 /api/v1，浏览器请求得自己带令牌。
   * 首次可用 `/?token=…` 打开（读走后立刻从地址栏抹掉，别留在浏览历史里），之后存本机。
   * SSE 用不了请求头（EventSource 的限制），只能走 ?token= —— 服务端只对 /stream 认它。
   */
  function resolveToken() {
    var q = null;
    try { q = new URLSearchParams(location.search).get("token"); } catch (e) { q = null; }
    if (q) {
      try {
        localStorage.setItem("governed.token", q);
        history.replaceState(null, "", location.pathname + location.hash);
      } catch (e) { /* 无痕模式等：令牌只留在内存里，照样能用 */ }
      return q;
    }
    try { return localStorage.getItem("governed.token") || ""; } catch (e) { return ""; }
  }

  var state = {
    token: resolveToken(),
    currentId: localStorage.getItem("governed.currentId") || "",
    es: null,                // 当前 EventSource
    listTimer: null,         // 列表 / 审批角标轮询
    approvalCount: 0,
  };

  /* ---------------- 基础工具 ---------------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmtTime(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return esc(iso);
    var p = function (n) { return String(n).padStart(2, "0"); };
    return p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }
  function fmtClock(ts) {
    if (!ts) return "";
    var d = typeof ts === "number" ? new Date(ts) : new Date(ts);
    if (isNaN(d.getTime())) return "";
    var p = function (n) { return String(n).padStart(2, "0"); };
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }
  function fmtNum(n) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return Number(n).toLocaleString("zh-CN");
  }
  function pct(x) { return (x === null || x === undefined) ? "—" : Math.round(x * 100) + "%"; }

  function api(path, opts) {
    opts = opts || {};
    var init = { headers: {}, method: opts.method || "GET" };
    if (state.token) init.headers["Authorization"] = "Bearer " + state.token;
    if (opts.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }
    return fetch(API + path, init).then(function (r) {
      return r.text().then(function (txt) {
        var data = null;
        try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = { raw: txt }; }
        if (!r.ok) {
          var msg = (data && (data.detail || data.message)) || ("HTTP " + r.status);
          var err = new Error(msg);
          err.status = r.status; err.data = data;
          if (r.status === 401) showTokenPrompt(msg);
          throw err;
        }
        return data;
      });
    });
  }

  /* 令牌缺失 / 失效时的入口：外壳是匿名取到的，令牌只能由人填一次。 */
  function showTokenPrompt(msg) {
    var c = $("#content");
    if (!c || $("#tok")) return;
    c.innerHTML =
      '<div class="empty"><span class="big">⚿</span>' + esc(msg || "需要访问令牌") +
      '<div class="row" style="max-width:440px;margin:14px auto 0">' +
        '<input class="input grow" id="tok" type="password" placeholder="粘贴 AUTH_TOKENS 里的令牌" />' +
        '<button class="btn primary" id="tok-ok">保存并重载</button></div>' +
      '<div class="small dim" style="margin-top:10px">令牌只存在本机浏览器；也可以用 <code>/?token=…</code> 打开本页。</div></div>';
    var save = function () {
      var v = $("#tok").value.trim();
      if (!v) return;
      try { localStorage.setItem("governed.token", v); } catch (e) {}
      location.reload();
    };
    $("#tok-ok").onclick = save;
    $("#tok").addEventListener("keydown", function (e) { if (e.key === "Enter") save(); });
  }

  function toast(msg, kind) {
    var box = $("#toast");
    if (!box) return;
    var t = el("div", "banner " + (kind || "info"), esc(msg));
    t.style.cssText = "position:fixed;right:18px;bottom:18px;z-index:50;min-width:240px;max-width:420px;box-shadow:0 6px 24px rgba(0,0,0,.18)";
    box.appendChild(t);
    setTimeout(function () { t.remove(); }, 4200);
  }

  /* ---------------- 状态 → chip ---------------- */
  var STATUS_META = {
    finished:      ["ok", "✓", "完成"],
    completed:     ["ok", "✓", "完成"],
    running:       ["info", "▶", "运行中"],
    in_progress:   ["warn", "▶", "进行中"],
    awaiting_approval: ["warn", "⚑", "待审批"],
    blocked:       ["warn", "⏸", "阻塞"],
    pending:       ["idle", "○", "待开始"],
    skipped:       ["idle", "↷", "跳过"],
    cancelled:     ["idle", "✕", "已取消"],
    failed:        ["bad", "✕", "失败"],
    active:        ["ok", "✓", "已装载"],
    loading:       ["warn", "▶", "装载中"],
    disposed:      ["idle", "○", "已卸载"],
    unloaded:      ["idle", "○", "未装载"],
  };
  function chip(status, label) {
    var m = STATUS_META[status] || ["idle", "○", status || "未知"];
    return '<span class="chip ' + m[0] + '"><span class="sym">' + m[1] + '</span>' + esc(label || m[2]) + "</span>";
  }
  function led(on, kind) {
    return '<span class="led ' + (on ? (kind || "on") : "off") + '"></span>';
  }
  function boolChip(v, onText, offText) {
    return v
      ? '<span class="chip ok"><span class="sym">✓</span>' + esc(onText || "开") + "</span>"
      : '<span class="chip idle"><span class="sym">○</span>' + esc(offText || "关") + "</span>";
  }

  /* ---------------- 路由 ---------------- */
  var ROUTES = {
    tasks: { title: "任务列表", render: renderTasks },
    detail: { title: "运行详情", render: renderDetail, contextual: true },
    plan: { title: "计划审批", render: renderPlan, contextual: true },
    approvals: { title: "审批台", render: renderApprovals },
    control: { title: "管控面", render: renderControl },
    metrics: { title: "运行时指标", render: renderMetrics, contextual: true },
    artifacts: { title: "工作区 / 产物", render: renderArtifacts, contextual: true },
    packages: { title: "插件 / 领域包", render: renderPackages },
  };

  function parseHash() {
    var h = (location.hash || "#/tasks").replace(/^#\/?/, "");
    var parts = h.split("/");
    return { name: parts[0] || "tasks", id: parts[1] ? decodeURIComponent(parts[1]) : "" };
  }
  function go(hash) { location.hash = hash; }

  function setCurrentId(id) {
    state.currentId = id || "";
    if (id) localStorage.setItem("governed.currentId", id);
    else localStorage.removeItem("governed.currentId");
  }

  function route() {
    closeStream();
    stopListPolling();
    var r = parseHash();
    var conf = ROUTES[r.name] || ROUTES.tasks;
    if (r.id) setCurrentId(r.id);
    var id = r.id || state.currentId;

    document.querySelectorAll("#nav .item").forEach(function (a) {
      a.classList.toggle("active", a.getAttribute("data-route") === (ROUTES[r.name] ? r.name : "tasks"));
    });
    $("#page-title").textContent = conf.title;
    $("#page-crumb").textContent = conf.contextual && id ? "会话 " + id.slice(0, 8) : "";
    $("#task-context").textContent = conf.contextual ? (id ? "当前会话：" + id.slice(0, 8) : "未选择会话") : "";

    $("#content").innerHTML = '<div class="empty"><span class="big">◷</span>加载中…</div>';
    Promise.resolve()
      .then(function () { return conf.render(id, r); })
      .catch(function (e) {
        // 401 已经渲染成"填令牌"页了，别用报错横幅把它盖掉
        if (e && e.status === 401) return;
        $("#content").innerHTML =
          '<div class="banner err"><strong>加载失败：</strong>' + esc(e.message) + "</div>";
      });
  }

  function needId(id) {
    if (!id) {
      $("#content").innerHTML =
        '<div class="empty"><span class="big">▤</span>请先在「任务列表」选择一个会话。<br><br>' +
        '<button class="btn primary" id="go-tasks">去任务列表</button></div>';
      $("#go-tasks").onclick = function () { go("#/tasks"); };
      return false;
    }
    return true;
  }

  /* ---------------- 页 1：任务列表 ---------------- */
  function renderTasks() {
    var c = $("#content");
    c.innerHTML =
      '<div class="toolbar">' +
        '<input class="input grow" id="goal" placeholder="描述任务目标，回车创建…（自然语言）" />' +
        '<input class="input" id="ctx" style="width:220px" placeholder="补充上下文（可选）" />' +
        '<button class="btn primary" id="create">＋ 新建任务</button>' +
        '<span class="spacer"></span>' +
        '<button class="btn ghost sm" id="refresh">刷新</button>' +
      "</div>" +
      '<div class="panel"><div class="panel-body flush"><table class="grid">' +
        "<thead><tr><th>状态</th><th>任务目标</th><th>进度</th><th>子任务</th><th>更新时间</th><th></th></tr></thead>" +
        '<tbody id="rows"><tr><td colspan="6" class="empty">加载中…</td></tr></tbody>' +
      "</table></div></div>";

    function create() {
      var goal = $("#goal").value.trim();
      if (!goal) { toast("请填写任务目标", "warn"); return; }
      var context = $("#ctx").value.trim();
      var btn = $("#create");
      btn.disabled = true;
      api("/tasks", { method: "POST", body: { goal: goal, context: context } })
        .then(function (r) {
          setCurrentId(r.thread_id);
          go("#/detail/" + encodeURIComponent(r.thread_id));
        })
        .catch(function (e) { toast("创建失败：" + e.message, "err"); btn.disabled = false; });
    }
    $("#create").onclick = create;
    $("#goal").addEventListener("keydown", function (e) { if (e.key === "Enter") create(); });
    $("#refresh").onclick = load;

    function load() {
      return api("/tasks").then(function (rows) {
        var body = $("#rows");
        if (!body) return;
        if (!rows.length) {
          body.innerHTML = '<tr><td colspan="6" class="empty"><span class="big">▤</span>还没有任务，在上方新建一个。</td></tr>';
          return;
        }
        rows.sort(function (a, b) { return (b.updated_at || "").localeCompare(a.updated_at || ""); });
        body.innerHTML = rows.map(function (t) {
          var action = t.awaiting_approval
            ? '<button class="btn sm danger" data-act="approve">去审批</button>'
            : '<button class="btn sm" data-act="open">查看</button>';
          return '<tr class="clickable" data-id="' + esc(t.thread_id) + '">' +
            "<td>" + chip(t.status) + (t.awaiting_approval ? ' <span class="chip warn"><span class="sym">⚑</span>待你处理</span>' : "") + "</td>" +
            '<td style="max-width:420px"><div>' + esc(t.goal) + "</div>" +
              (t.error ? '<div class="small" style="color:var(--bad-ink)">' + esc(t.error) + "</div>" : "") +
              (t.has_final ? '<div class="small dim">已产出最终结论</div>' : "") + "</td>" +
            '<td style="min-width:120px"><div style="display:flex;align-items:center;gap:8px"><div class="progress"><i style="width:' + pct(t.progress) + '"></i></div><span class="small mono">' + pct(t.progress) + "</span></div></td>" +
            '<td class="nowrap small mono">' + t.task_completed + " / " + t.task_total + "</td>" +
            '<td class="nowrap small dim">' + fmtTime(t.updated_at) + "</td>" +
            '<td class="right">' + action + "</td>" +
          "</tr>";
        }).join("");
        body.querySelectorAll("tr").forEach(function (tr) {
          tr.onclick = function (e) {
            var id = tr.getAttribute("data-id");
            setCurrentId(id);
            var btnAct = e.target.closest && e.target.closest("button");
            if (btnAct && btnAct.getAttribute("data-act") === "approve") { go("#/approvals"); }
            else { go("#/detail/" + encodeURIComponent(id)); }
          };
        });
      }).catch(function (e) {
        var body = $("#rows");
        if (body) body.innerHTML = '<tr><td colspan="6" class="empty" style="color:var(--bad-ink)">' + esc(e.message) + "</td></tr>";
      });
    }
    load();
    state.listTimer = setInterval(load, 4000);
  }

  /* ---------------- 共用：拉一个任务状态 ---------------- */
  function fetchStatus(id) { return api("/tasks/" + encodeURIComponent(id)); }

  function stepsHtml(plan) {
    if (!plan || !plan.tasks || !plan.tasks.length) {
      return '<div class="empty">计划尚未生成（规划节点还没跑完）。</div>';
    }
    return '<ol class="steps" style="padding-left:4px">' + plan.tasks.map(function (t) {
      var st = t.status === "completed" ? "ok"
        : t.status === "in_progress" ? "warn"
        : t.status === "failed" ? "bad"
        : t.status === "awaiting_approval" ? "warn"
        : t.status === "skipped" || t.status === "cancelled" ? "idle" : "idle";
      var sym = st === "ok" ? "✓" : st === "bad" ? "✕" : st === "warn" ? "▶" : "○";
      var gate = t.gate_decision && t.gate_decision !== "approved" && t.gate_decision !== "pass"
        ? '<div class="gate"><strong>质量门：</strong>' + esc(t.gate_decision) +
          (t.gate_note ? " — " + esc(t.gate_note) : "") + "</div>" : "";
      var retry = (t.retry_count && t.retry_count > 0)
        ? ' <span class="chip warn"><span class="sym">↻</span>重试 ' + t.retry_count + "</span>" : "";
      return '<li class="step"><div class="rail"><span class="node ' + st + '">' + sym + "</span><span class=\"bar\"></span></div>" +
        '<div class="body">' +
          '<div class="title">' + esc(t.title || t.task_id) + " " + chip(t.status) + retry +
            ' <span class="chip info mono"><span class="sym">⬡</span>' + esc(t.assigned_to || "未分配") + "</span></div>" +
          '<div class="meta">' + (t.description ? esc(t.description) : "") + "</div>" +
          (t.depends_on && t.depends_on.length ? '<div class="meta">依赖：' + t.depends_on.map(esc).join("、") + "</div>" : "") +
          (t.expected_artifacts && t.expected_artifacts.length ? '<div class="meta">预期产物：' + t.expected_artifacts.map(esc).join("、") + "</div>" : "") +
          gate +
        "</div></li>";
    }).join("") + "</ol>";
  }

  /* ---------------- 页 2：计划审批（执行前；当前为只读计划视图） ---------------- */
  function renderPlan(id) {
    if (!needId(id)) return;
    var c = $("#content");
    c.innerHTML =
      '<div class="banner warn"><strong>执行前计划审批开关当前未启用。</strong>' +
      "本页只读呈现 <code>TaskPlan</code>；运行过程中的「工具审批」与「质量门 HUMAN」请在审批台处理。</div>" +
      '<div class="panel"><div class="panel-head"><h2>任务计划</h2></div><div class="panel-body" id="plan-body"><div class="empty">加载中…</div></div></div>';
    fetchStatus(id).then(function (s) {
      var plan = s.plan;
      if (!plan) { $("#plan-body").innerHTML = '<div class="empty">该会话还没有计划。</div>'; return; }
      var rows = (plan.tasks || []).map(function (t) {
        return "<tr>" +
          "<td>" + esc(t.task_id) + "</td>" +
          "<td>" + esc(t.title || "") + '<div class="small dim">' + esc(t.description || "") + "</div></td>" +
          "<td>" + chip(t.status) + "</td>" +
          '<td class="mono small">' + esc(t.assigned_to || "") + "</td>" +
          '<td class="small">' + (t.depends_on || []).map(esc).join("、") + "</td>" +
          '<td class="small">' + (t.acceptance_criteria || []).map(esc).join("；") + "</td>" +
          '<td class="small">' + (t.expected_artifacts || []).map(esc).join("、") + "</td>" +
        "</tr>";
      }).join("");
      $("#plan-body").innerHTML =
        '<dl class="kv" style="margin-bottom:14px">' +
          "<dt>目标</dt><dd>" + esc(plan.goal) + "</dd>" +
          "<dt>版本 / 重规划</dt><dd class='mono'>v" + esc(plan.version) + " · replan " + fmtNum(plan.replan_count) + " · 进度 " + pct(plan.progress) + "</dd>" +
        "</dl>" +
        '<div class="panel-body flush"><table class="grid"><thead><tr>' +
          "<th>ID</th><th>子任务</th><th>状态</th><th>分配给</th><th>依赖</th><th>验收标准</th><th>预期产物</th>" +
        "</tr></thead><tbody>" + rows + "</tbody></table></div>";
    });
  }

  /* ---------------- 页 3：审批台 ---------------- */
  function renderApprovals() {
    var c = $("#content");
    c.innerHTML =
      '<div class="toolbar"><span class="muted">列出所有停在「待审批」的会话（工具审批 / 质量门 HUMAN）。</span>' +
      '<span class="spacer"></span><button class="btn ghost sm" id="refresh-appr">刷新</button></div>' +
      '<div id="appr-list"><div class="empty">加载中…</div></div>';
    $("#refresh-appr").onclick = load;

    function approvalCard(id, a) {
      var p = a.payload || {};
      var kind = p.kind || p.type || "tool";
      var kindChip = kind === "gate"
        ? '<span class="chip info"><span class="sym">⛨</span>质量门审查</span>'
        : '<span class="chip warn"><span class="sym">⚑</span>工具审批</span>';
      var title = p.task_title || p.tool || p.description || "待审批动作";
      var args = p.arguments || p.arguments_preview || p.code || null;
      var dl = [
        ["请求编号", a.interrupt_id],
        ["工具", p.tool],
        ["子 Agent", p.sub_agent || p.agent],
        ["子任务", p.task_id || p.task_title],
        ["沙箱任务", p.sandbox_task],
        ["过期时间", p.expires_at ? fmtTime(p.expires_at) : null],
      ].filter(function (kv) { return kv[1]; })
        .map(function (kv) { return "<dt>" + esc(kv[0]) + '</dt><dd class="mono">' + esc(kv[1]) + "</dd>"; }).join("");
      var card = el("div", "approval");
      // "本任务内不再询问"只对**工具审批**有意义：豁免的键是 (会话, 工具)，质量门没有
      // 工具这一维（服务端也会丢弃它的 remember）。所以 gate 卡片不给这两个按钮 ——
      // 给了就是点了没用。
      var rememberRow = kind === "gate" ? "" :
        '<div class="row" style="margin-top:6px">' +
          '<span class="small dim" style="flex:1">本任务内不再询问 <code class="mono">' +
            esc(p.tool || "该动作") + '</code>：</span>' +
          '<button class="btn ghost sm" data-act="always-allow">总是允许</button>' +
          '<button class="btn ghost sm" data-act="always-deny">总是拒绝</button></div>';
      card.innerHTML =
        '<div class="h">' + kindChip + "<strong>" + esc(title) + "</strong>" +
        '<span class="spacer"></span><span class="small dim">会话 ' + esc(id.slice(0, 8)) + "</span></div>" +
        (p.description ? '<div class="small muted">' + esc(p.description) + "</div>" : "") +
        (dl ? '<dl class="kv" style="margin-top:8px">' + dl + "</dl>" : "") +
        (args ? "<pre>" + esc(typeof args === "string" ? args : JSON.stringify(args, null, 2)) + "</pre>" : "") +
        '<div class="row"><input class="input grow" placeholder="审批意见（驳回时建议填写原因）" />' +
        '<button class="btn success" data-act="approve">✓ 批准执行</button>' +
        '<button class="btn danger" data-act="reject">✕ 驳回</button></div>' +
        rememberRow;
      var input = card.querySelector("input");
      card.querySelector('[data-act="approve"]').onclick = function () { decide(id, true, input.value, card, null); };
      card.querySelector('[data-act="reject"]').onclick = function () { decide(id, false, input.value, card, null); };
      ['always-allow', 'always-deny'].forEach(function (act) {
        var b = card.querySelector('[data-act="' + act + '"]');
        if (b) b.onclick = function () {
          decide(id, act === "always-allow", input.value, card,
                 act === "always-allow" ? "allow" : "deny");
        };
      });
      return card;
    }

    function decide(id, approved, comment, card, remember) {
      card.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
      api("/tasks/" + encodeURIComponent(id) + "/approval", {
        method: "POST",
        body: { approved: approved, comment: comment, remember: remember || null },
      })
        .then(function () { toast(approved ? "已批准，任务继续执行" : "已驳回", "info"); load(); })
        .catch(function (e) {
          toast("审批提交失败：" + e.message, "err");
          card.querySelectorAll("button").forEach(function (b) { b.disabled = false; });
        });
    }

    function load() {
      api("/tasks").then(function (rows) {
        var pending = rows.filter(function (t) { return t.awaiting_approval; });
        state.approvalCount = pending.length;
        updateApprovalBadge();
        var box = $("#appr-list");
        if (!pending.length) { box.innerHTML = '<div class="empty"><span class="big">✓</span>当前没有待处理的审批。</div>'; return; }
        box.innerHTML = "";
        pending.forEach(function (t) {
          api("/tasks/" + encodeURIComponent(t.thread_id) + "/approvals").then(function (items) {
            var wrap = el("div");
            wrap.appendChild(el("div", "section-title", esc(t.goal)));
            if (!items.length) {
              wrap.appendChild(el("div", "small dim", "状态标记为待审批但未取到中断项，可刷新或查看运行详情。"));
            } else {
              items.forEach(function (a) { wrap.appendChild(approvalCard(t.thread_id, a)); });
            }
            box.appendChild(wrap);
          });
        });
      }).catch(function (e) {
        if (e && e.status === 401) return;   // 已渲染成"填令牌"页
        var box = $("#appr-list");
        if (box) box.innerHTML = '<div class="banner err">' + esc(e.message) + "</div>";
      });
    }
    load();
    state.listTimer = setInterval(load, 6000);
  }

  function updateApprovalBadge() {
    var badge = $("#nav-approval-badge"), num = $("#nav-approval-count");
    if (!badge) return;
    if (state.approvalCount > 0) {
      badge.classList.remove("hidden"); num.textContent = state.approvalCount;
    } else { badge.classList.add("hidden"); }
  }

  /* ---------------- 页 4：运行详情（子任务状态机 + SSE 时间线 + 审批卡） ---------------- */
  var EV_META = {
    RUN_STARTED: ["info", "运行开始"], RUN_FINISHED: ["ok", "运行完成"], RUN_ERROR: ["bad", "运行出错"],
    SUBAGENT_STARTED: ["info", "子 Agent 开始"], SUBAGENT_FINISHED: ["ok", "子 Agent 结束"],
    TOOL_CALL_START: ["idle", "工具调用"], TOOL_CALL_END: ["info", "工具返回"],
    GUARD_DECISION: ["bad", "管控裁决"], APPROVAL_REQUIRED: ["warn", "提请审批"], APPROVAL_RESOLVED: ["ok", "审批结论"],
    STATE_SNAPSHOT: ["idle", "状态快照"], METRICS: ["idle", "指标"], SPAN_ENTER: ["idle", "进入"], SPAN_EXIT: ["idle", "退出"],
  };
  function evSummary(type, d) {
    d = d.data || d;
    switch (type) {
      case "TOOL_CALL_START": return (d.tool || "") + (d.agent ? " · " + d.agent : "");
      case "TOOL_CALL_END": return (d.tool || "") + (d.ok ? " ✓" : " ✕") + (d.duration_ms != null ? " · " + d.duration_ms + "ms" : "") + (d.cache_hit ? " · 缓存命中" : "");
      case "SUBAGENT_STARTED": return (d.agent || "") + " — " + (d.title || d.task_id || "");
      case "SUBAGENT_FINISHED": return (d.agent || "") + (d.gate_decision ? " · 门：" + d.gate_decision : "") + (d.duration_ms != null ? " · " + d.duration_ms + "ms" : "");
      case "GUARD_DECISION": return (d.layer || "") + " / " + (d.decision || "") + (d.reason ? " — " + d.reason : "") + (d.tool ? " · " + d.tool : "");
      case "APPROVAL_REQUIRED": return (d.kind || "tool") + " · " + (d.tool || d.task_title || "");
      case "APPROVAL_RESOLVED": return (d.approved ? "批准" : "驳回") + (d.expired ? "（已过期）" : "") + (d.approver ? " · " + d.approver : "");
      case "RUN_FINISHED": return d.final_answer ? "已产出最终结论" : "";
      case "RUN_ERROR": return d.error || "";
      default: return d.tool || d.agent || d.title || "";
    }
  }

  function renderDetail(id) {
    if (!needId(id)) return;
    var c = $("#content");
    c.innerHTML =
      '<div class="tiles" id="head-tiles"></div>' +
      '<div id="grants"></div>' +
      '<div class="panel"><div class="panel-head"><h2>子任务状态机</h2><span class="spacer"></span>' +
        '<span class="small dim" id="plan-progress"></span></div><div class="panel-body" id="steps"><div class="empty">加载中…</div></div></div>' +
      '<div class="panel hidden" id="approval-panel"><div class="panel-head"><h2>待审批</h2></div><div class="panel-body" id="approvals"></div></div>' +
      '<div class="panel"><div class="panel-head"><h2>结论 / 输出</h2></div><div class="panel-body" id="final"></div></div>' +
      '<div class="panel"><div class="panel-head"><h2>实时事件时间线</h2><span class="spacer"></span>' +
        '<span class="small"><span class="live-dot" id="live"></span> <span id="live-text">连接中…</span></span></div>' +
        '<div class="panel-body flush"><div class="timeline" id="timeline"></div></div></div>';

    var timeline = $("#timeline");
    function appendEv(e) {
      var meta = EV_META[e.type] || ["idle", e.type];
      var d = e.data || {};
      var row = el("div", "ev");
      row.innerHTML =
        '<div class="t">' + esc(fmtClock(e.ts) || fmtClock(d.ts)) + "</div>" +
        '<div class="tag"><span class="chip ' + meta[0] + ' mono"><span class="sym">●</span>' + esc(e.type) + "</span></div>" +
        '<div class="d">' + esc(evSummary(e.type, e)) + "</div>";
      timeline.appendChild(row);
      timeline.scrollTop = timeline.scrollHeight;
    }

    function paint(s) {
      var tiles = $("#head-tiles");
      tiles.innerHTML =
        tile("状态", chip(s.status), "") +
        tile("计划进度", pct(s.progress), s.plan ? (s.plan.tasks || []).length + " 个子任务" : "") +
        tile("待审批", s.pending_approvals && s.pending_approvals.length ? String(s.pending_approvals.length) : "0", s.pending_approvals && s.pending_approvals.length ? "需要你处理" : "无") +
        tile("Token 调用", s.token_usage ? fmtNum(s.token_usage.calls) : "—", s.token_usage ? "总 " + fmtNum(s.token_usage.total) : "");
      $("#steps").innerHTML = stepsHtml(s.plan);
      $("#plan-progress").textContent = s.plan ? ("v" + s.plan.version + " · replan " + s.plan.replan_count) : "";
      var ap = $("#approval-panel");
      if (s.pending_approvals && s.pending_approvals.length) {
        ap.classList.remove("hidden");
        $("#approvals").innerHTML = "";
        s.pending_approvals.forEach(function (a) {
          var p = a.payload || {};
          var args = p.arguments || p.code || null;
          var card = el("div", "approval");
          card.innerHTML =
            '<div class="h"><strong>' + esc(p.task_title || p.tool || "待审批动作") + "</strong><span class=\"spacer\"></span>" +
              '<span class="small dim mono">' + esc(a.interrupt_id) + "</span></div>" +
            (p.description ? '<div class="small muted">' + esc(p.description) + "</div>" : "") +
            (args ? "<pre>" + esc(typeof args === "string" ? args : JSON.stringify(args, null, 2)) + "</pre>" : "") +
            '<div class="row"><input class="input grow" placeholder="审批意见（可选）" />' +
            '<button class="btn success" data-a="1">✓ 批准</button><button class="btn danger" data-a="0">✕ 驳回</button></div>';
          var input = card.querySelector("input");
          card.querySelector('[data-a="1"]').onclick = function () { decide(s.thread_id, true, input.value); };
          card.querySelector('[data-a="0"]').onclick = function () { decide(s.thread_id, false, input.value); };
          $("#approvals").appendChild(card);
        });
      } else { ap.classList.add("hidden"); }

      // 本任务内已生效的审批豁免：一次静默放宽，必须一眼可见（不是藏在卡片里的开关）
      var grantsBox = $("#grants");
      if (grantsBox) {
        var grants = s.session_grants || [];
        grantsBox.innerHTML = grants.length
          ? '<div class="banner info"><strong>本任务内已豁免：</strong>' + grants.map(function (g) {
              return '<code class="mono">' + esc(g.tool) + "</code>（" +
                (g.effect === "allow" ? "不再询问，直接放行" : "不再询问，直接拒绝") +
                (g.granted_by ? "，由 " + esc(g.granted_by) + " 授予" : "") + "）";
            }).join("、") + "</div>"
          : "";
      }
      $("#final").innerHTML =
        s.final_answer ? '<pre class="json" style="white-space:pre-wrap;font-family:var(--sans);max-height:none">' + esc(s.final_answer) + "</pre>"
        : s.error ? '<div class="banner err">' + esc(s.error) + "</div>"
        : '<div class="dim small">尚无最终结论（任务运行结束后在此呈现）。</div>';
    }
    function decide(tid, approved, comment) {
      api("/tasks/" + encodeURIComponent(tid) + "/approval", { method: "POST", body: { approved: approved, comment: comment } })
        .then(function () { toast(approved ? "已批准" : "已驳回", "info"); })
        .catch(function (e) { toast("审批失败：" + e.message, "err"); });
    }

    fetchStatus(id).then(paint).catch(function (e) {
      $("#steps").innerHTML = '<div class="banner err">' + esc(e.message) + "</div>";
    });

    // SSE：事件流追加时间线；status / approval 消息刷新权威快照
    var es = new EventSource(API + "/tasks/" + encodeURIComponent(id) + "/stream" +
      (state.token ? "?token=" + encodeURIComponent(state.token) : ""));
    state.es = es;
    $("#live").classList.add("on"); $("#live-text").textContent = "实时连接";
    es.addEventListener("open", function () { $("#live").classList.add("on"); $("#live-text").textContent = "实时连接"; });
    es.addEventListener("status", function (ev) { try { paint(JSON.parse(ev.data)); } catch (e) {} });
    es.addEventListener("approval", function () { fetchStatus(id).then(paint); });
    es.addEventListener("final", function (ev) { try { var d = JSON.parse(ev.data); if (d.final_answer) fetchStatus(id).then(paint); } catch (e) {} });
    es.addEventListener("done", function () { $("#live").classList.remove("on"); $("#live-text").textContent = "已结束"; es.close(); });
    es.addEventListener("error", function (ev) {
      // 浏览器会自动重连；服务端在任务终态后关闭流属正常
      if (ev && ev.data) { try { var d = JSON.parse(ev.data); if (d.error) toast(d.error, "err"); } catch (e) {} }
      $("#live-text").textContent = "重连中…";
    });
    // 其余具名事件（RUN_*/TOOL_*/SUBAGENT_*/GUARD_*/APPROVAL_*/SPAN_* 等）统一追加
    Object.keys(EV_META).forEach(function (type) {
      es.addEventListener(type, function (ev) {
        try { appendEv(JSON.parse(ev.data)); } catch (e) {}
      });
    });
  }

  function tile(label, valueHtml, sub) {
    return '<div class="tile"><div class="label">' + esc(label) + '</div><div class="value">' + valueHtml +
      '</div>' + (sub ? '<div class="sub">' + esc(sub) + "</div>" : "") + "</div>";
  }

  /* ---------------- 页 5：管控面 ---------------- */
  function renderControl() {
    var c = $("#content");
    c.innerHTML = '<div class="empty">加载中…</div>';
    api("/control-plane").then(function (cp) {
      var tools = cp.tools || {};
      var toolRows = (tools.tools || []).map(function (t) {
        return "<tr>" +
          '<td class="mono">' + esc(t.name) + "</td>" +
          '<td><span class="chip info mono"><span class="sym">⬡</span>' + esc(t.required_role) + "</span></td>" +
          '<td class="mono small">' + fmtNum(t.rate_limit_per_min) + " / min</td>" +
          "<td>" + (t.requires_approval ? '<span class="chip warn"><span class="sym">⚑</span>需审批</span>' : '<span class="chip idle"><span class="sym">○</span>免审</span>') + "</td>" +
          "<td>" + (t.run_in_sandbox ? '<span class="chip info"><span class="sym">⛨</span>沙箱</span>' : '<span class="chip idle"><span class="sym">○</span>本机</span>') + "</td>" +
          '<td class="mono small right">' + fmtNum(t.recent_calls_1min) + "</td>" +
        "</tr>";
      }).join("");

      function sw(name, desc, on, onText, offText) {
        return '<div class="switch-row"><div class="nm">' + esc(name) + "</div><div class=\"ds\">" + esc(desc) +
          '</div><div class="switch-state">' + led(on, on ? "on" : "off") + " " +
          esc(on ? (onText || "已启用") : (offText || "未启用")) + "</div></div>";
      }
      var perm = cp.permission || {}, permRt = perm.runtime || {};
      var sbx = cp.sandbox || {}, cb = cp.circuit_breaker || {}, cache = cp.cache || {};
      var mwItems = (cp.middleware && cp.middleware.items) || [];
      // 限流没有全局开关，配额是逐工具声明的：有配额的工具有几个，才是这一行的真实状态。
      var rateLimited = (tools.tools || []).filter(function (t) { return (t.rate_limit_per_min || 0) > 0; }).length;

      c.innerHTML =
        '<div class="tiles">' +
          tile("注册工具", fmtNum(tools.total_tools), "见下方工具管控表") +
          tile("检查点后端", esc((cp.checkpointer && cp.checkpointer.backend) || "—"), "会话持久化") +
          tile("长期记忆", esc((cp.memory && cp.memory.backend) || "—"), cp.memory && cp.memory.degraded_reason ? "已降级" : "") +
          tile("PDP 默认策略", esc(permRt.effective_default_policy || perm.default_policy || "—"), "无权工具的默认处置") +
          tile("审批超时", fmtNum((cp.approval || {}).timeout_seconds) + " s", "超时只能驳回") +
          tile("限流作用域", esc((cp.rate_limit || {}).scope || "—"), "") +
        "</div>" +

        '<div class="panel"><div class="panel-head"><h2>管控组件开关（配置 vs 运行时）</h2></div><div class="panel-body flush">' +
          sw("鉴权 Auth", "请求身份与角色来源", (cp.auth || {}).configured_enabled) +
          sw("PDP 权限判定", "存在性 + 角色 / 子 Agent 白名单", permRt.pdp_enabled, "运行中", "未运行") +
          sw("行列级权限", "SQL 改写 / 行级过滤（与 PDP 共用 PERMISSION_* 开关）", perm.configured_enabled, "已接入", "未启用") +
          sw("熔断器", "错误率 / 超阈自动断路", cb.configured_enabled, JSON.stringify(cb.snapshot || {}) !== "{}" ? "有快照" : "已启用", "未启用") +
          sw("限流", "按角色 / 工具的每分钟配额（" + ((cp.rate_limit || {}).scope || "process") + " 作用域）", rateLimited > 0, rateLimited + " 个工具设了配额", "无工具设配额") +
          sw("沙箱", "高危工具隔离执行（网络：" + (sbx.network_enabled ? "开" : "关") + "）", sbx.configured_enabled, sbx.connected ? "已连接" : "未连接", "未启用") +
          sw("PII 脱敏", "工具结果中的敏感信息", (cp.pii || {}).configured_enabled) +
          sw("结果缓存", "幂等只读工具命中缓存", cache.configured_enabled) +
          sw("审计", "JSONL + Kafka 尽力投递", !!tools.audit_enabled, "运行中", "未接入") +
        "</div></div>" +

        '<div class="panel"><div class="panel-head"><h2>中间件链</h2></div><div class="panel-body flush">' +
          (mwItems.length ? '<table class="grid"><thead><tr><th>顺序</th><th>名称</th><th>状态</th></tr></thead><tbody>' +
            mwItems.map(function (m) {
              return "<tr><td class='mono'>" + fmtNum(m.priority) + "</td><td class='mono'>" + esc(m.name) + "</td><td>" + boolChip(m.enabled, "启用", "停用") + "</td></tr>";
            }).join("") + "</tbody></table>" : '<div class="empty">无已注册中间件</div>') +
        "</div></div>" +

        '<div class="panel"><div class="panel-head"><h2>工具管控表</h2><span class="spacer"></span>' +
          '<span class="small dim">需审批：' + ((cp.approval || {}).requires_approval_tools || []).join("、") + "</span></div>" +
          '<div class="panel-body flush"><table class="grid"><thead><tr>' +
          "<th>工具</th><th>最低角色</th><th>限流</th><th>审批</th><th>沙箱</th><th class='right'>近 1 分钟调用</th>" +
          "</tr></thead><tbody>" + toolRows + "</tbody></table></div></div>" +

        '<div class="panel"><div class="panel-head"><h2>数据源 / 缓存</h2></div><div class="panel-body">' +
          '<dl class="kv">' +
            "<dt>数据源</dt><dd>" + esc(((cp.datasources || {}).names || []).join("、") || "—") + "</dd>" +
            "<dt>缓存命中 / 未命中</dt><dd class='mono'>" + fmtNum(cache.runtime && cache.runtime.hits) + " / " + fmtNum(cache.runtime && cache.runtime.misses) +
              "（容量 " + fmtNum(cache.runtime && cache.runtime.entries) + "/" + fmtNum(cache.runtime && cache.runtime.max_entries) + "）</dd>" +
            "<dt>沙箱工具</dt><dd>" + esc((sbx.sandboxed_tools || []).join("、") || "—") + "</dd>" +
          "</dl></div></div>";
    });
  }

  /* ---------------- 页 6：运行时指标 ---------------- */
  function renderMetrics(id) {
    if (!needId(id)) return;
    var c = $("#content");
    c.innerHTML = '<div class="empty">加载中…</div>';
    api("/tasks/" + encodeURIComponent(id) + "/metrics").then(function (m) {
      var a = m.audit || {}, g = m.guard_events || {}, ctx = m.context || {}, cache = m.cache || {}, tokens = m.tokens || {};
      var byToolRows = Object.keys(a.by_tool || {}).map(function (name) {
        var b = a.by_tool[name];
        return "<tr><td class='mono'>" + esc(name) + "</td><td class='right mono'>" + b.calls + "</td>" +
          '<td class="right mono" style="color:var(--bad-ink)">' + b.failed + "</td>" +
          '<td class="right mono" style="color:var(--warn-ink)">' + b.denied + "</td></tr>";
      }).join("");
      var guardRows = Object.keys(g.guard_by_layer || {}).map(function (layer) {
        var b = g.guard_by_layer[layer] || {};
        return "<tr><td class='mono'>" + esc(layer) + "</td>" +
          '<td class="right mono" style="color:var(--ok-ink)">' + (b.allow || 0) + "</td>" +
          '<td class="right mono" style="color:var(--bad-ink)">' + (b.deny || 0) + "</td>" +
          '<td class="right mono" style="color:var(--warn-ink)">' + (b.defer || 0) + "</td></tr>";
      }).join("");
      var totals = ctx.totals || {};
      var cr = cache.runtime || {}, tu = tokens.usage || {};
      c.innerHTML =
        '<div class="banner info">口径不同，已逐项标注作用域：审计为<b>持久化本地 JSONL（重启保留）</b>；' +
        "管控事件为<b>进程内自启动累计</b>；缓存 / Token 为组件单例的<b>进程累计</b>。</div>" +
        '<div class="tiles">' +
          tile("工具调用总数", fmtNum(a.total), a.scope) +
          tile("成功 / 失败", fmtNum(a.succeeded) + " / " + fmtNum(a.failed), "审计") +
          tile("PDP 拒绝", fmtNum(a.pdp_denied), "权限判定") +
          tile("管控事件", fmtNum(g.guard_total), g.scope) +
          tile("审批提请 / 已决", fmtNum(g.approval_required) + " / " +
            fmtNum((g.approval_resolved || {}).approved + (g.approval_resolved || {}).rejected + (g.approval_resolved || {}).expired), "批准/驳回/过期") +
          tile("缓存命中", fmtNum(a.cache_hits), "审计口径") +
          tile("沙箱执行", fmtNum(a.sandbox_used), "") +
          tile("平均时延", fmtNum(a.duration_ms_avg) + " ms", "每次工具调用") +
        "</div>" +

        '<div class="panel"><div class="panel-head"><h2>按管控层分类的裁决（进程内）</h2></div><div class="panel-body flush">' +
          (guardRows ? '<table class="grid"><thead><tr><th>管控层</th><th class="right">放行</th><th class="right">拒绝</th><th class="right"> defer</th></tr></thead><tbody>' + guardRows + "</tbody></table>"
            : '<div class="empty">本次进程内该会话没有管控拦截 / 改写事件（常态放行不计数）。</div>') +
          '<div class="panel-body small dim">审批结论：批准 ' + fmtNum((g.approval_resolved || {}).approved) +
            " · 驳回 " + fmtNum((g.approval_resolved || {}).rejected) +
            " · 过期 " + fmtNum((g.approval_resolved || {}).expired) + "</div>" +
        "</div>" +

        '<div class="panel"><div class="panel-head"><h2>按工具的调用（持久审计）</h2></div><div class="panel-body flush">' +
          (byToolRows ? '<table class="grid"><thead><tr><th>工具</th><th class="right">调用</th><th class="right">失败</th><th class="right">PDP 拒绝</th></tr></thead><tbody>' + byToolRows + "</tbody></table>"
            : '<div class="empty">该会话暂无审计到的工具调用。</div>') +
        "</div></div>" +

        '<div class="panel"><div class="panel-head"><h2>上下文 / 缓存 / Token</h2></div><div class="panel-body">' +
          '<dl class="kv">' +
            "<dt>大结果沉淀</dt><dd>" + fmtNum(ctx.settled_count) + " 个文件引用（该会话）</dd>" +
            "<dt>沉淀字符（原始/回注）</dt><dd class='mono'>" + fmtNum(totals.sink_chars_original) + " / " + fmtNum(totals.sink_chars_returned) +
              " · 截断 " + fmtNum(totals.truncate_count) + " · 压缩折叠 " + fmtNum(totals.compact_messages_folded) + " 条</dd>" +
            "<dt>预算超限</dt><dd class='mono'>" + fmtNum(totals.budget_violation_count) + " 次（进程累计）</dd>" +
            "<dt>缓存</dt><dd class='mono'>命中 " + fmtNum(cr.hits) + " · 未命中 " + fmtNum(cr.misses) + " · 容量 " + fmtNum(cr.entries) + "/" + fmtNum(cr.max_entries) + "（进程累计）</dd>" +
            "<dt>Token</dt><dd class='mono'>调用 " + fmtNum(tu.calls) + " · 提示 " + fmtNum(tu.prompt) + " · 补全 " + fmtNum(tu.completion) + " · 总 " + fmtNum(tu.total) + "（进程累计）</dd>" +
          "</dl></div></div>";
    });
  }

  /* ---------------- 页 7：工作区 / 产物 ---------------- */
  function renderArtifacts(id) {
    if (!needId(id)) return;
    var c = $("#content");
    c.innerHTML = '<div class="empty">加载中…</div>';
    api("/tasks/" + encodeURIComponent(id) + "/artifacts").then(function (a) {
      var taskRows = (a.task_artifacts || []).map(function (t) {
        var expected = (t.expected_artifacts || []).map(esc).join("、") || "—";
        var art = t.artifacts && Object.keys(t.artifacts).length
          ? "<pre class='json'>" + esc(JSON.stringify(t.artifacts, null, 2)) + "</pre>"
          : '<span class="dim small">暂无回填产物索引</span>';
        return "<tr><td style='min-width:200px'><strong>" + esc(t.title || t.task_id) + "</strong>" +
          '<div class="small dim">' + esc(t.task_id || "") + " · " + esc(t.assigned_to || "") + "</div></td>" +
          "<td>" + chip(t.status) + "</td><td class='small'>" + expected + "</td><td style='min-width:260px'>" + art + "</td></tr>";
      }).join("");
      var settled = (a.settled_refs || []).map(function (r) {
        return "<tr><td class='mono'>" + esc(r.tool || "") + "</td><td class='mono'>" + esc(r.path || "") +
          "</td><td class='right mono'>" + fmtNum(r.chars) + " 字符</td></tr>";
      }).join("");
      var vfs = (a.vfs_root || []).map(function (f) {
        var typeChip = f.type === "directory" ? '<span class="chip info"><span class="sym">▸</span>目录</span>'
          : '<span class="chip idle"><span class="sym">▪</span>文件</span>';
        return "<tr><td>" + typeChip + "</td><td class='mono'>" + esc(f.path || f.name) + "</td><td class='right mono'>" + fmtNum(f.size) + " B</td></tr>";
      }).join("");
      c.innerHTML =
        '<div class="banner info">控制台只按<b>产出物类型</b>与框架的产物契约渲染，不猜测领域报告结构；' +
        "大结果全文沉淀在 VFS，图状态里只保留路径 / 类型等标量索引。</div>" +
        '<div class="panel"><div class="panel-head"><h2>子任务产物契约与回填</h2></div><div class="panel-body flush">' +
          '<table class="grid"><thead><tr><th>子任务</th><th>状态</th><th>预期产物</th><th>实际产物索引</th></tr></thead><tbody>' +
          (taskRows || '<tr><td colspan="4" class="empty">无</td></tr>') + "</tbody></table>" +
        "</div></div>" +
        '<div class="panel"><div class="panel-head"><h2>大结果沉淀文件（上下文外溢 → VFS）</h2></div><div class="panel-body flush">' +
          (settled ? '<table class="grid"><thead><tr><th>来源工具</th><th>VFS 路径</th><th class="right">体量</th></tr></thead><tbody>' + settled + "</tbody></table>"
            : '<div class="empty">没有因超阈值而沉淀的大结果。</div>') +
        "</div></div>" +
        '<div class="panel"><div class="panel-head"><h2>虚拟文件系统根目录</h2></div><div class="panel-body flush">' +
          (vfs ? '<table class="grid"><thead><tr><th>类型</th><th>路径</th><th class="right">大小</th></tr></thead><tbody>' + vfs + "</tbody></table>"
            : '<div class="empty">VFS 为空。</div>') +
        "</div></div>";
    });
  }

  /* ---------------- 页 8：插件 / 领域包 ---------------- */
  function renderPackages() {
    var c = $("#content");
    c.innerHTML =
      '<div class="toolbar"><span class="muted">框架登记的领域包、声明式贡献与真实装载状态。</span>' +
      '<span class="spacer"></span><button class="btn ghost sm" id="pk-refresh">刷新</button></div><div id="pk" class="empty">加载中…</div>';
    $("#pk-refresh").onclick = load;
    function load() {
      api("/packages").then(function (rows) {
        if (!rows.length) { $("#pk").className = "empty"; $("#pk").innerHTML = '<span class="big">⬡</span>尚未登记任何领域包。'; return; }
        $("#pk").className = "";
        $("#pk").innerHTML = rows.map(function (p) {
          var contributes = p.contributes && Object.keys(p.contributes).length
            ? "<pre class='json'>" + esc(JSON.stringify(p.contributes, null, 2)) + "</pre>"
            : '<span class="dim small">无声明式贡献</span>';
          return '<div class="panel"><div class="panel-head"><h2>' + esc(p.name) + "</h2>" +
            '<span class="chip mono" style="margin-left:8px">v' + esc(p.version || "—") + "</span>" +
            '<span class="spacer"></span>' + chip(p.state) + "</div>" +
            '<div class="panel-body">' +
            (p.description ? '<div class="muted">' + esc(p.description) + "</div>" : "") +
            '<dl class="kv" style="margin-top:10px">' +
              "<dt>提供者</dt><dd>" + esc(p.provider || "—") + "</dd>" +
              "<dt>框架服务依赖</dt><dd>" + ((p.requires || []).map(esc).join("、") || "—") + "</dd>" +
              "<dt>贡献摘要</dt><dd>" + esc(p.contributes_summary || "—") + "</dd>" +
              (p.status_note ? "<dt>装载说明</dt><dd>" + esc(p.status_note) + "</dd>" : "") +
              (p.error ? "<dt>错误</dt><dd style='color:var(--bad-ink)'>" + esc(p.error) + "</dd>" : "") +
            "</dl>" + contributes + "</div></div>";
        }).join("");
      }).catch(function (e) { $("#pk").innerHTML = '<div class="banner err">' + esc(e.message) + "</div>"; });
    }
    load();
  }

  /* ---------------- SSE / 轮询生命周期 ---------------- */
  function closeStream() {
    if (state.es) { try { state.es.close(); } catch (e) {} state.es = null; }
  }
  function stopListPolling() {
    if (state.listTimer) { clearInterval(state.listTimer); state.listTimer = null; }
  }

  /* ---------------- 健康检查 + 审批角标（全局） ---------------- */
  function health() {
    fetch("/health").then(function (r) { return r.json(); }).then(function (h) {
      $("#health-dot").className = "dot ok";
      $("#health-text").textContent = "服务在线";
      $("#health-version").textContent = "v" + (h.version || "");
    }).catch(function () {
      $("#health-dot").className = "dot bad";
      $("#health-text").textContent = "服务不可达";
    });
  }
  function pollApprovalBadge() {
    api("/tasks").then(function (rows) {
      var n = rows.filter(function (t) { return t.awaiting_approval; }).length;
      state.approvalCount = n; updateApprovalBadge();
    }).catch(function () {});
  }

  /* ---------------- 启动 ---------------- */
  window.addEventListener("hashchange", route);
  document.querySelectorAll("#nav .item").forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      var name = a.getAttribute("data-route");
      var contextual = (ROUTES[name] || {}).contextual;
      if (contextual && state.currentId) { go("#/" + name + "/" + encodeURIComponent(state.currentId)); }
      else { go("#/" + name); }
    });
  });
  health();
  setInterval(health, 15000);
  pollApprovalBadge();
  setInterval(pollApprovalBadge, 8000);
  if (!location.hash) location.hash = "#/tasks";
  route();
})();
