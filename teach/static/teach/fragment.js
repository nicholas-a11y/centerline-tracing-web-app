// Parse the optional "#s=..." fragment the main app puts on the Teach link.
// Format: base64url(UTF-8 JSON) of {"v": 1, "settings": {all ten settings}, "image_sha256": "<64 hex>"}
// ("image_sha256" optional). The fragment never leaves the browser; the server re-validates
// every value it receives. Anything malformed, oversized, out of range or with extra or
// missing keys returns null, and the page silently keeps its defaults.
(function (root) {
  "use strict";
  var MAX_FRAGMENT_CHARS = 2048;
  var TOP_KEYS = { v: true, settings: true, image_sha256: true };

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
    var m = /^#?s=([A-Za-z0-9_-]+)$/.exec(hash);
    if (!m) return null;
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
    return { settings: out, imageSha256: sha };
  }

  var api = { parseSettingsFragment: parseSettingsFragment, MAX_FRAGMENT_CHARS: MAX_FRAGMENT_CHARS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TeachFragment = api;
})(typeof window !== "undefined" ? window : this);
