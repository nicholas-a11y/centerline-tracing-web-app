// Manage page: the token lives in the URL fragment (#k=...), which browsers never send to the server.
(function () {
  "use strict";
  var root = document.getElementById("manage");
  var subId = root.getAttribute("data-sub-id");
  var status = document.getElementById("manage-status");
  var body = document.getElementById("manage-body");
  var facts = document.getElementById("manage-facts");
  var overlay = document.getElementById("manage-overlay");
  var answerForm = document.getElementById("answer-form");
  var deleteForm = document.getElementById("delete-form");
  var match = /(?:^#|&)k=([^&]+)/.exec(window.location.hash);
  var token = match ? decodeURIComponent(match[1]) : "";
  var api = "/teach/api/manage/" + encodeURIComponent(subId);

  function call(method, payload) {
    var opts = { method: method, headers: { "X-Teach-Manage-Token": token }, credentials: "same-origin" };
    if (payload) opts.body = payload;
    return fetch(api, opts).then(function (resp) {
      return resp.json().then(function (data) { data.httpStatus = resp.status; return data; });
    });
  }

  function errorText(data) {
    var e = (data.errors || [])[0];
    return e ? e.message + (e.hint ? " " + e.hint : "") : "Request failed (HTTP " + data.httpStatus + ").";
  }

  function add(term, value) {
    if (value === undefined || value === null || value === "") return;
    var dt = document.createElement("dt");
    dt.textContent = term;
    var dd = document.createElement("dd");
    dd.textContent = String(value);
    facts.appendChild(dt);
    facts.appendChild(dd);
  }

  function render(r) {
    facts.textContent = "";
    add("Number", "#" + r.number);
    add("Status", (r.status || "").replace(/_/g, " "));
    add("Title", r.title);
    add("Public title", r.title_hidden ? "(hidden by the maintainer)" : r.title_public);
    add("What is wrong", r.what_is_wrong);
    add("What you expected", r.what_is_expected);
    add("Notes", r.notes);
    add("Maintainer note", r.admin_note_public);
    add("Me too", r.me_too_count);
    var last = (r.results || [])[r.results.length - 1];
    if (last) add("Latest engine outcome", last.outcome + " (engine " + last.engine_commit + ")");
    if (r.overlay_png && r.overlay_png.indexOf("data:image/png;base64,") === 0) {
      overlay.src = r.overlay_png;
      overlay.hidden = false;
    }
    answerForm.hidden = !r.can_answer;
    body.hidden = false;
  }

  function load() {
    if (!token) {
      status.textContent = "This link has no manage token. Use the full link you saved when you submitted.";
      return;
    }
    call("GET").then(function (data) {
      if (!data.ok) { status.textContent = errorText(data); return; }
      status.textContent = "";
      render(data.result);
    }).catch(function () { status.textContent = "Network error; please reload."; });
  }

  answerForm.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var fd = new FormData(answerForm);
    status.textContent = "Sending…";
    call("PATCH", fd).then(function (data) {
      status.textContent = data.ok ? "Answer sent." : errorText(data);
      if (data.ok) load();
    }).catch(function () { status.textContent = "Network error; please try again."; });
  });

  deleteForm.addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (!document.getElementById("delete-confirm").checked) {
      status.textContent = "Tick the box to confirm deletion.";
      return;
    }
    call("DELETE").then(function (data) {
      if (data.ok) {
        body.hidden = true;
        status.textContent = "Deleted. Only the number remains, marked deleted.";
      } else {
        status.textContent = errorText(data);
      }
    }).catch(function () { status.textContent = "Network error; please try again."; });
  });

  load();
})();
