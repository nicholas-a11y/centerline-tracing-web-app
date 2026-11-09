// Parse the optional "#s=...[&sid=...]" fragment the main app puts on the Teach link.
// "s" is base64url(UTF-8 JSON) of {"v": 1, "settings": {all ten settings}, "image_sha256": "<64 hex>",
// "crop": {"top", "bottom", "left", "right"}} ("image_sha256" and "crop" optional; crop values are
// the app's fractions 0..1). "sid" is the main app's session id (a lowercase UUID) and needs
// "image_sha256". The fragment never leaves the browser: the page only uses sid in the path of
// GET /teach/api/crop/<sid>/image.png, and the server re-validates every value it receives.
// Anything malformed, oversized, out of range or with extra or missing keys returns null, and
// the page silently keeps its defaults.
(function (root) {
  "use strict";
  var MAX_FRAGMENT_CHARS = 2048;
  var TOP_KEYS = { v: true, settings: true, image_sha256: true, crop: true };
  var CROP_KEYS = ["top", "bottom", "left", "right"];
  var SID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

  function decodeBase64Url(text) {
    if (!/^[A-Za-z0-9_-]+$/.test(text)) return null;
    var b64 = text.replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    try {
      var bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch (e) {
      return null;
    }
  }

  function checkValue(spec, v) {
    if (spec.kind === "boolean") return typeof v === "boolean";
    if (spec.kind === "string") return typeof v === "string" && spec.options.indexOf(v) >= 0;
    if (typeof v !== "number" || !isFinite(v)) return false;
    if (spec.kind === "integer" && Math.floor(v) !== v) return false;
    return v >= spec.min && v <= spec.max;
  }

  // specs: {name: {kind: "boolean"|"string"|"number"|"integer", min, max, options}}
  function parseSettingsFragment(hash, specs) {
    if (typeof hash !== "string" || hash.length > MAX_FRAGMENT_CHARS) return null;
    var m = /^#?s=([A-Za-z0-9_-]+)(?:&sid=([^&]*))?$/.exec(hash);
    if (!m) return null;
    var sid = m[2] === undefined ? null : m[2];
    if (sid !== null && !SID_RE.test(sid)) return null;
    var text = decodeBase64Url(m[1]);
    if (text === null) return null;
    var doc;
    try { doc = JSON.parse(text); } catch (e) { return null; }
    if (!doc || typeof doc !== "object" || Array.isArray(doc) || doc.v !== 1) return null;
    var k;
    for (k in doc) if (Object.prototype.hasOwnProperty.call(doc, k) && !TOP_KEYS[k]) return null;
    var s = doc.settings;
    if (!s || typeof s !== "object" || Array.isArray(s)) return null;
    var names = Object.keys(specs);
    if (Object.keys(s).length !== names.length) return null;
    var out = {};
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      if (!Object.prototype.hasOwnProperty.call(s, name) || !checkValue(specs[name], s[name])) return null;
      out[name] = s[name];
    }
    var sha = null;
    if (doc.image_sha256 !== undefined) {
      if (typeof doc.image_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(doc.image_sha256)) return null;
      sha = doc.image_sha256;
    }
    var crop = null;
    if (doc.crop !== undefined) {
      var c = doc.crop;
      if (!c || typeof c !== "object" || Array.isArray(c) || Object.keys(c).length !== CROP_KEYS.length) return null;
      crop = {};
      for (var j = 0; j < CROP_KEYS.length; j++) {
        var v = c[CROP_KEYS[j]];
        if (typeof v !== "number" || !isFinite(v) || v < 0 || v > 1) return null;
        crop[CROP_KEYS[j]] = v;
      }
    }
    if (sid !== null && sha === null) return null;
    return { settings: out, imageSha256: sha, crop: crop, sid: sid };
  }

  // Inverse of parseSettingsFragment, used by the main app's "Send this to Teach" link.
  function buildSettingsFragment(settings, imageSha256, crop, sid) {
    var doc = { v: 1, settings: settings };
    if (imageSha256) doc.image_sha256 = imageSha256;
    if (crop) {
      doc.crop = {};
      CROP_KEYS.forEach(function (k) { doc.crop[k] = Number(crop[k]); });
    }
    var bytes = new TextEncoder().encode(JSON.stringify(doc));
    var bin = "";
    for (var i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    var b64 = typeof btoa === "function" ? btoa(bin) : Buffer.from(bin, "binary").toString("base64");
    var hash = "#s=" + b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    return sid ? hash + "&sid=" + sid : hash;
  }

  var api = { parseSettingsFragment: parseSettingsFragment, buildSettingsFragment: buildSettingsFragment,
              MAX_FRAGMENT_CHARS: MAX_FRAGMENT_CHARS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TeachFragment = api;
})(typeof window !== "undefined" ? window : this);
