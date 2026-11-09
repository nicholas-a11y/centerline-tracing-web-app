// Main app's Teach checkbox (Stage 5c item 4). This file is only included in the main page
// when Teach is registered, and it reveals the checkbox only if GET /teach/api/status
// returns 200. With the box unticked it does nothing beyond recording which crop and
// settings the last run used; no request it makes changes anything on the server.
(function () {
  "use strict";
  var panel = document.getElementById("teachPanel");
  if (!panel || typeof fetch !== "function") return;
  var toggle = document.getElementById("teachToggle");
  var tools = document.getElementById("teachTools");
  var downloadBtn = document.getElementById("teachDownloadCropBtn");
  var sendWrap = document.getElementById("teachSendWrap");
  var sendLink = document.getElementById("teachSendLink");
  var note = document.getElementById("teachNote");

  // The ten Teach settings (teach/schema.py SETTINGS), taken from the app's own getParameters().
  var SETTING_KEYS = ["dark_threshold", "min_path_length", "enable_pruning", "normalization_mode",
    "normalization_sensitivity", "enable_post_fit_export", "source_smoothing", "enable_curve_fitting",
    "cubic_fit_tolerance", "force_orthogonal_as_lines"];

  var pendingRun = null;
  var lastRun = null;
  var refreshToken = 0;

  function sessionId() {
    // currentSessionId is a top-level `let` in the main page's script.
    return typeof currentSessionId !== "undefined" ? currentSessionId : null;
  }

  function cropQuery(crop) {
    if (!crop) return "";
    return "?" + ["top", "bottom", "left", "right"].map(function (k) {
      return k + "=" + encodeURIComponent(String(Number(crop[k])));
    }).join("&");
  }

  function currentCrop() {
    return typeof window.getCropParams === "function" ? window.getCropParams() : null;
  }

  function showNote(text) {
    note.textContent = text || "";
    note.hidden = !text;
  }

  function effectiveSettings(params, info) {
    var out = {};
    SETTING_KEYS.forEach(function (k) { out[k] = params[k]; });
    // The image was normalised with the session's settings at upload; send those.
    out.normalization_mode = info.normalization_mode;
    out.normalization_sensitivity = info.normalization_sensitivity;
    return out;
  }

  function refresh() {
    var token = ++refreshToken;
    tools.hidden = !toggle.checked;
    sendWrap.hidden = true;
    showNote("");
    if (!toggle.checked || !lastRun || lastRun.sid !== sessionId()) return;
    fetch("/teach/api/crop/" + encodeURIComponent(lastRun.sid) + cropQuery(lastRun.crop),
          { credentials: "same-origin", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (info) {
        if (token !== refreshToken || !info) return;
        if (!info.equivalent) {
          showNote(info.note);
          return;
        }
        var settings = effectiveSettings(lastRun.params, info);
        // The session id and crop let the Teach page fetch this same cropped image (pre-attach).
        sendLink.href = "/teach" + window.TeachFragment.buildSettingsFragment(
          settings, info.image_sha256, lastRun.crop, lastRun.sid);
        sendWrap.hidden = false;
      })
      .catch(function () {});
  }

  function download() {
    var sid = sessionId();
    if (!sid) {
      showNote("Upload an image first.");
      return;
    }
    // After a run, download exactly what that run processed; before one, the current crop.
    var crop = lastRun && lastRun.sid === sid ? lastRun.crop : currentCrop();
    fetch("/teach/api/crop/" + encodeURIComponent(sid) + "/image.png" + cropQuery(crop),
          { credentials: "same-origin", cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("download failed");
        var name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "");
        return r.blob().then(function (blob) { return { blob: blob, name: name ? name[1] : "cropped.png" }; });
      })
      .then(function (out) {
        var url = URL.createObjectURL(out.blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = out.name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
      })
      .catch(function () { showNote("The cropped image could not be downloaded. Upload the image again."); });
  }

  function wrap(name, before, after) {
    var original = window[name];
    if (typeof original !== "function") return;
    window[name] = function () {
      if (before) before.apply(this, arguments);
      var result = original.apply(this, arguments);
      if (after) after.apply(this, arguments);
      return result;
    };
  }

  function init() {
    // processImage snapshots the session, crop and settings it sends to /process_immediate;
    // displayImmediateResults marks that run as shown; clearResults forgets it.
    wrap("processImage", function () {
      pendingRun = {
        sid: sessionId(),
        crop: currentCrop(),
        params: typeof window.getParameters === "function" ? window.getParameters() : {},
      };
    });
    wrap("displayImmediateResults", null, function () {
      if (pendingRun) {
        lastRun = pendingRun;
        pendingRun = null;
        refresh();
      }
    });
    wrap("clearResults", null, function () {
      lastRun = null;
      refresh();
    });
    toggle.addEventListener("change", refresh);
    downloadBtn.addEventListener("click", download);
    toggle.checked = false;
    panel.hidden = false;
  }

  fetch("/teach/api/status", { credentials: "same-origin", cache: "no-store" })
    .then(function (r) { if (r.status === 200) init(); })
    .catch(function () {});
})();
