// Teach submission page. All output is built with textContent; nothing from a response is parsed as HTML.
(function () {
  "use strict";
  var form = document.getElementById("case-form");
  var out = document.getElementById("result");

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = String(text);
    if (cls) e.className = cls;
    return e;
  }

  function val(id) { return document.getElementById(id).value.trim(); }
  function checked(id) { return document.getElementById(id).checked; }

  var settingInputs = Array.prototype.slice.call(document.querySelectorAll("[data-setting]"));
  var settingDefaults = {};
  settingInputs.forEach(function (inp) {
    settingDefaults[inp.dataset.setting] = inp.type === "checkbox" ? inp.defaultChecked :
      inp.tagName === "SELECT" ? inp.querySelector("option[selected]").value : inp.defaultValue;
  });

  function settingLabel(inp) { return document.querySelector("label[for='" + inp.id + "']").textContent.replace(/\s*\(.*\)\s*$/, "").trim(); }

  // Every setting is always sent explicitly: the server's own fallbacks differ from the app's defaults.
  function readSettings() {
    var s = {};
    settingInputs.forEach(function (inp) {
      var kind = inp.dataset.kind;
      if (kind === "boolean") s[inp.dataset.setting] = inp.checked;
      else if (kind === "string") s[inp.dataset.setting] = inp.value;
      else s[inp.dataset.setting] = inp.value.trim() === "" ? NaN : Number(inp.value);
    });
    return s;
  }

  function writeSettings(values) {
    settingInputs.forEach(function (inp) {
      var v = values[inp.dataset.setting];
      if (v === undefined) return;
      if (inp.dataset.kind === "boolean") inp.checked = !!v; else inp.value = String(v);
    });
    updateSettingsLine();
  }

  function updateSettingsLine() {
    var s = readSettings();
    document.getElementById("settings-used").textContent = settingInputs.map(function (inp) {
      var v = s[inp.dataset.setting];
      return settingLabel(inp) + " " + (inp.dataset.kind === "boolean" ? (v ? "on" : "off") : String(v));
    }).join("; ");
  }

  function buildMeta() {
    var meta = {
      title: val("title"), description: val("description"),
      consent: checked("consent"), public_thumbnail: checked("public_thumbnail"), share_bitmap: checked("share_bitmap"),
      settings: readSettings()
    };
    if (val("not_duplicate_reason")) meta.not_duplicate_reason = val("not_duplicate_reason");
    return meta;
  }

  function buildBody() {
    var fd = new FormData();
    var img = document.getElementById("image").files[0];
    var ref = document.getElementById("reference").files[0];
    if (img) fd.append("image", img);
    if (ref) fd.append("reference", ref);
    fd.append("meta", JSON.stringify(buildMeta()));
    return fd;
  }

  function problemList(title, items, cls) {
    var wrap = el("div");
    if (!items || !items.length) return wrap;
    wrap.appendChild(el("h3", title));
    var ul = el("ul");
    items.forEach(function (p) {
      var li = el("li", null, cls);
      li.appendChild(el("strong", p.code + " (" + p.field + "): "));
      li.appendChild(document.createTextNode(p.message + (p.hint ? " " + p.hint : "")));
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    return wrap;
  }

  function showResult(body, status) {
    out.textContent = "";
    if (body.ok) {
      out.appendChild(el("p", "Request is valid.", "pass"));
    } else {
      out.appendChild(el("p", "Request has problems (HTTP " + status + ").", "error"));
    }
    out.appendChild(problemList("Errors", body.errors, "error"));
    out.appendChild(problemList("Warnings", body.warnings, "warn"));
    if (body.number) {
      out.appendChild(el("h3", "Submitted as #" + body.number));
      out.appendChild(el("p", "Save your manage link now. It is shown only once and cannot be recovered:"));
      var code = el("code", body.manage_url, "link");
      out.appendChild(code);
      var q = el("p");
      var a = el("a", "View #" + body.number + " in the queue");
      a.href = body.queue_url;
      q.appendChild(a);
      out.appendChild(q);
    }
    var r = body.result;
    if (r && r.outcome) {
      var passed = r.outcome === "pass";
      out.appendChild(el("h3", "Engine result"));
      out.appendChild(el("p", passed ? "PASS: the engine already handles this case." :
        "Outcome: " + r.outcome.toUpperCase() + (r.outcome === "fail" ? " (the engine gets this wrong; a good case to submit)" : ""),
        passed ? "pass" : "fail"));
      var dl = el("dl");
      var s = r.summary || {};
      [["Registration", r.registration], ["Paths", s.path_count], ["Junctions in region", s.junctions_in_region],
       ["Lowest single-path coverage", s.min_single_coverage], ["Worst p95 error (px)", s.max_p95_px],
       ["Engine commit", r.engine_commit], ["Time (ms)", r.elapsed_ms]].forEach(function (kv) {
        if (kv[1] === undefined || kv[1] === null) return;
        dl.appendChild(el("dt", kv[0]));
        dl.appendChild(el("dd", kv[1]));
      });
      out.appendChild(dl);
      if (r.lines && r.lines.length) {
        var ul = el("ul");
        r.lines.forEach(function (ln) {
          ul.appendChild(el("li", "Line " + ln.index + ": " + (ln.passed ? "pass" : "fail") +
            ", coverage " + ln.best_single_coverage + ", p95 " + ln.p95_px + " px" + (ln.reason ? " (" + ln.reason + ")" : "")));
        });
        out.appendChild(ul);
      }
      if (r.overlay_png && r.overlay_png.indexOf("data:image/png;base64,") === 0) {
        var img = el("img");
        img.src = r.overlay_png;
        img.alt = "Overlay: engine centerlines in red, your reference in green";
        out.appendChild(img);
      }
    }
    if (body.result && body.result.matches) {
      out.appendChild(el("h3", body.result.matches.length ? "Similar items" : "No similar items found"));
      var ml = el("ul");
      body.result.matches.forEach(function (m) {
        var li = el("li");
        var a = el("a", "#" + m.number + (m.title ? ": " + m.title : ""));
        a.href = m.url;
        li.appendChild(a);
        li.appendChild(document.createTextNode(" (" + m.status + ")"));
        ml.appendChild(li);
      });
      out.appendChild(ml);
      if (body.result.exact_duplicate) out.appendChild(el("p", "This exact case is already in the queue.", "error"));
    }
    out.focus();
  }

  function send(url, label) {
    out.textContent = label + "…";
    fetch(url, { method: "POST", body: buildBody(), credentials: "same-origin" })
      .then(function (resp) {
        return resp.json().then(function (body) { showResult(body, resp.status); },
          function () { out.textContent = "Unexpected response (HTTP " + resp.status + ")."; });
      })
      .catch(function () { out.textContent = "Network error; please try again."; });
  }

  out.tabIndex = -1;
  document.getElementById("btn-validate").addEventListener("click", function () { send("/teach/api/validate", "Validating and scoring"); });
  document.getElementById("btn-similar").addEventListener("click", function () { send("/teach/api/similar", "Checking"); });
  document.getElementById("btn-submit").addEventListener("click", function () {
    if (!checked("consent")) {
      out.textContent = "Please accept the consent and licence text first.";
      return;
    }
    send("/teach/api/submissions", "Submitting");
  });
  form.addEventListener("submit", function (ev) { ev.preventDefault(); });
  settingInputs.forEach(function (inp) { inp.addEventListener("change", updateSettingsLine); });
  document.getElementById("settings-reset").addEventListener("click", function () { writeSettings(settingDefaults); });
  updateSettingsLine();
})();
