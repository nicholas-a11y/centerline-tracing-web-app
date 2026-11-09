(function () {
  "use strict";
  var token = sessionStorage.getItem("teach-admin-token") || "";
  var message = document.getElementById("admin-message");
  var content = document.getElementById("admin-content");
  var form = document.getElementById("admin-signin");
  var body = document.getElementById("admin-body");
  function node(tag, text, parent) {
    var el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (parent) parent.appendChild(el);
    return el;
  }
  async function call(path, method, data) {
    var opts = {method: method || "GET", headers: {"Authorization": "Bearer " + token}, cache: "no-store"};
    if (data !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(data);
    }
    var resp = await fetch("/teach/api/admin/" + path, opts);
    if (resp.status === 401 || resp.status === 429) {
      body.hidden = true; form.hidden = false;
      throw new Error(resp.status === 429 ? "Too many wrong tries. Wait 15 minutes." : "The token is not valid.");
    }
    var result = await resp.json();
    if (!resp.ok || result.ok === false) throw new Error(result.error || "Request failed.");
    return result;
  }
  function button(text, action, parent) {
    var el = node("button", text, parent);
    el.type = "button";
    el.addEventListener("click", async function () {
      el.disabled = true;
      try { await action(); } catch (e) { message.textContent = e.message; }
      finally { el.disabled = false; }
    });
    return el;
  }
  function facts(data, parent) {
    var dl = node("dl", undefined, parent);
    Object.keys(data).forEach(function (key) {
      node("dt", key.replace(/_/g, " "), dl);
      node("dd", typeof data[key] === "object" ? JSON.stringify(data[key]) : String(data[key]), dl);
    });
  }
  async function environment() {
    var data = await call("environment");
    content.textContent = "";
    node("h2", "Environment", content);
    facts(data, content);
    button("Test storage write/read/delete", async function () {
      await call("storage-check", "POST"); message.textContent = "Storage check passed.";
    }, content);
    button("Send test alert", async function () {
      var r = await call("test-alert", "POST"); message.textContent = r.message;
    }, content);
    button("Rebuild index", async function () {
      var r = await call("rebuild-index", "POST"); message.textContent = "Index rebuilt: " + r.items + " items.";
    }, content);
  }
  var labels = {pending: "Awaiting review", accepted: "Accepted", rejected: "Rejected"};
  async function queue(filter, query) {
    var data = await call("cases?" + new URLSearchParams({status: filter || "", q: query || ""}));
    content.textContent = "";
    node("h2", "Review queue", content);
    var cards = node("div", undefined, content); cards.className = "admin-cards";
    Object.keys(labels).forEach(function (s) {
      button(labels[s] + ": " + data.counts[s], function () { return queue(s, query); }, cards);
    });
    button("All: " + data.total, function () { return queue("", query); }, cards);
    var search = node("form", undefined, content);
    var searchLabel = node("label", "Search title or number", search); searchLabel.htmlFor = "admin-search";
    var input = node("input", undefined, search); input.id = "admin-search"; input.value = query || ""; input.maxLength = 120;
    var submit = node("button", "Search", search); submit.type = "submit";
    search.addEventListener("submit", function (ev) {
      ev.preventDefault(); queue(filter, input.value).catch(function (e) { message.textContent = e.message; });
    });
    var wrap = node("div", undefined, content); wrap.className = "admin-table";
    var table = node("table", undefined, wrap);
    node("caption", "Submitted cases", table);
    var head = node("tr", undefined, node("thead", undefined, table));
    ["No.", "Title", "Status", "Result now", "Failing since", "Me too", "Settings", "Date", "Actions"].forEach(function (t) {
      var th = node("th", t, head); th.scope = "col";
    });
    var tbody = node("tbody", undefined, table);
    data.items.forEach(function (item) {
      var tr = node("tr", undefined, tbody);
      [item.number, item.title + (item.title_hidden ? " (public title hidden)" : ""), labels[item.status] + (item.merged ? " / Merged" : ""),
        item.outcome === "pass" ? "Pass" : item.outcome === "fail" ? "Fail" : item.outcome,
        item.history_label || "Not recorded", item.me_too_count, item.settings, (item.created_at || "").slice(0, 10)].forEach(function (t) {
        node("td", String(t), tr);
      });
      var actions = node("td", undefined, tr);
      var link = node("a", item.status === "pending" ? "Review" : "Open", actions);
      link.href = "/teach/admin/cases/" + item.number;
      button(item.title_hidden ? "Unhide title" : "Hide title", async function () {
        await call("cases/" + item.number, "PATCH", {title_hidden: !item.title_hidden}); await queue(filter, query);
      }, actions);
    });
    node("p", "Showing " + data.shown + " of " + data.total + ". Last retention sweep: " +
      (data.retention ? data.retention.finished_at : "Not run yet"), content);
  }
  function field(form, name, label, value, type) {
    var p = node("p", undefined, form);
    var lab = node("label", label, p); lab.htmlFor = "review-" + name;
    var el = node(type === "textarea" ? "textarea" : "input", undefined, p);
    el.id = "review-" + name; el.name = name;
    if (type !== "textarea") el.type = type || "text";
    if (type === "checkbox") el.checked = !!value; else el.value = value === null || value === undefined ? "" : value;
    if (type === "textarea") { el.rows = 3; el.maxLength = 4000; }
    return el;
  }
  async function download(path, filename) {
    var resp = await fetch("/teach/api/admin/" + path, {headers: {Authorization: "Bearer " + token}, cache: "no-store"});
    if (!resp.ok) throw new Error("Download failed (HTTP " + resp.status + ").");
    var url = URL.createObjectURL(await resp.blob());
    var a = node("a"); a.href = url; a.download = filename; a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }
  async function casePage(number) {
    var r = (await call("cases/" + number)).case;
    content.textContent = "";
    node("h2", "#" + r.number + ": " + r.title, content);
    var link = node("a", "Back to review queue", content); link.href = "/teach/admin";
    if (r.overlay_png) { var im = node("img", undefined, content); im.src = r.overlay_png; im.alt = "App centerlines and green hand traced ideal over the bitmap"; }
    if (r.post_fit_overlay_png) { var post = node("img", undefined, content); post.src = r.post_fit_overlay_png; post.alt = "Post-fit result and hand traced ideal"; }
    var last = (r.results || [])[r.results.length - 1] || {};
    var summary = last.summary || {};
    node("h3", "Metrics", content);
    facts({result: last.outcome || "Unknown", path_count: summary.path_count,
      hand_traced_count: r.reference.paths, p95_distance_px: summary.max_p95_px,
      largest_gap_px: summary.largest_gap_px === undefined ? "Not recorded; re-score this case" : summary.largest_gap_px}, content);
    var lines = ((last.metrics || {}).prune_off || {}).lines || [];
    lines.forEach(function (line) { facts({line: line.index + 1, single_path_coverage: line.best_single_coverage,
      p95_distance_px: (line.ref_to_best || {}).p95, largest_gap_px: line.largest_gap_px}, content); });
    node("h3", "Submitter text", content); node("p", r.description || "(No text supplied)", content);
    facts({bitmap_size: r.image.width + " x " + r.image.height + " px",
      crop_note: "This is the bitmap as submitted. It may already be cropped; no full-image crop coordinates are retained.",
      consent: r.consent, visibility_consent: r.visibility, me_too_count: r.me_too_count}, content);
    node("h3", "Ten settings used", content); facts(r.params, content);
    node("h3", "Score history", content);
    node("p", r.history_label || "Not recorded", content); facts({entries: r.results}, content);
    var edit = node("form", undefined, content);
    var label = node("label", "Status", edit); label.htmlFor = "review-status";
    var select = node("select", undefined, edit); select.id = "review-status";
    Object.keys(labels).forEach(function (s) { var opt = node("option", labels[s], select); opt.value = s; opt.selected = s === r.status; });
    var pub = field(edit, "public-note", "Public note (shown in the queue and the submitter's case)", r.admin_note_public, "textarea");
    var priv = field(edit, "private-note", "Private note (admin only)", r.admin_note_private, "textarea");
    var tol = field(edit, "tolerance", "Tolerance (px)", r.scoring.tolerance_px, "number"); tol.min = "0.05"; tol.max = "100"; tol.step = "0.05";
    var region = field(edit, "region", "Optional region: JSON with x0, y0, x1, y1 (blank = whole bitmap)",
      r.scoring.region ? JSON.stringify(r.scoring.region) : "", "textarea");
    var hide = field(edit, "hide", "Hide public title", r.title_hidden, "checkbox");
    var thumb = field(edit, "thumbnail", "Show thumbnail (requires submitter permission)", r.visibility.thumbnail_public, "checkbox");
    thumb.disabled = !r.visibility.thumbnail_opt_in;
    var bitmap = field(edit, "bitmap", "Show full bitmap (requires submitter permission)", r.visibility.bitmap_public, "checkbox");
    bitmap.disabled = !r.visibility.bitmap_opt_in;
    node("button", "Save review", edit).type = "submit";
    edit.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      try {
        await call("cases/" + number, "PATCH", {status: select.value, admin_note_public: pub.value,
          admin_note_private: priv.value, tolerance_px: Number(tol.value), region: region.value.trim() ? JSON.parse(region.value) : null,
          title_hidden: hide.checked, thumbnail_public: thumb.checked, bitmap_public: bitmap.checked});
        message.textContent = "Review saved."; await casePage(number);
      } catch (e) { message.textContent = e.message; }
    });
    button("Download hand traced SVG (as text only)", function () { return download("cases/" + number + "/hand_traced.svg", "hand_traced-" + number + ".svg"); }, content);
    button("Delete this case", async function () {
      if (!window.confirm("Delete all files and text? Only a deleted-case tombstone remains.")) return;
      await call("cases/" + number, "DELETE"); location.assign("/teach/admin");
    }, content);
  }
  async function load() {
    var match = /^\/teach\/admin\/cases\/(\d+)$/.exec(location.pathname);
    if (match) await casePage(Number(match[1]));
    else if (location.pathname === "/teach/admin/environment") await environment();
    else await queue();
    form.hidden = true; body.hidden = false;
  }
  form.addEventListener("submit", async function (ev) {
    ev.preventDefault();
    token = document.getElementById("admin-token").value;
    document.getElementById("admin-token").value = "";
    try {
      await load(); sessionStorage.setItem("teach-admin-token", token); message.textContent = "";
    } catch (e) { message.textContent = e.message; }
  });
  document.getElementById("admin-signout").addEventListener("click", function () {
    token = ""; sessionStorage.removeItem("teach-admin-token");
    content.textContent = ""; body.hidden = true; form.hidden = false;
  });
  if (token) load().catch(function (e) { message.textContent = e.message; });
})();
