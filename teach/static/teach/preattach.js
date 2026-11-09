// Pre-attach the cropped bitmap handed over from the main app ("#s=...&sid=...").
// The session id is used only in the path of GET /teach/api/crop/<sid>/image.png; the fetched
// file is checked against the fragment's sha-256 before it is offered. Nothing here is trusted
// by the server: the bitmap is still uploaded through the normal form and re-validated.
(function (root) {
  "use strict";
  var UNAVAILABLE = "The app's copy of this image is no longer available. Attach the downloaded cropped image instead.";
  var MISMATCH = "The image fetched from the app is not the one you sent from it. Attach the downloaded cropped image instead.";
  var UNVERIFIABLE = "This browser cannot check the image fetched from the app. Attach the downloaded cropped image instead.";
  var MAX_BYTES = 4 * 1024 * 1024;
  var PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

  function cropQuery(crop) {
    if (!crop) return "";
    return "?" + ["top", "bottom", "left", "right"].map(function (k) {
      return k + "=" + encodeURIComponent(String(Number(crop[k])));
    }).join("&");
  }

  function cropImageUrl(sid, crop) {
    return "/teach/api/crop/" + encodeURIComponent(sid) + "/image.png" + cropQuery(crop);
  }

  // Width and height from a PNG's IHDR chunk, or null.
  function pngSize(bytes) {
    if (!bytes || bytes.length < 24) return null;
    for (var i = 0; i < 8; i++) if (bytes[i] !== PNG_SIGNATURE[i]) return null;
    if (String.fromCharCode(bytes[12], bytes[13], bytes[14], bytes[15]) !== "IHDR") return null;
    function u32(o) { return ((bytes[o] << 24) >>> 0) + (bytes[o + 1] << 16) + (bytes[o + 2] << 8) + bytes[o + 3]; }
    var w = u32(16), h = u32(20);
    return w > 0 && h > 0 ? { width: w, height: h } : null;
  }

  function hex(buf) {
    return Array.prototype.map.call(new Uint8Array(buf), function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
  }

  function fileName(disposition) {
    var m = /filename="([A-Za-z0-9._-]{1,80})"/.exec(disposition || "");
    return m ? m[1] : "cropped.png";
  }

  // Remove a hand-over fragment from the address bar so the session id is not left in
  // history or in a copied URL. Returns true if it removed one.
  function stripFragment(loc, history) {
    if (!loc || typeof loc.hash !== "string" || !/^#s=/.test(loc.hash) || !history || !history.replaceState) return false;
    history.replaceState(history.state, "", loc.pathname + loc.search);
    return true;
  }

  // handover: parseSettingsFragment's result. deps: {fetch(url, opts), digest(ArrayBuffer) -> Promise<ArrayBuffer>|null}.
  // Resolves to null (nothing to fetch), {ok: true, bytes, name, width, height} or {ok: false, reason, message}.
  function fetchCropped(handover, deps) {
    if (!handover || !handover.sid || !handover.imageSha256) return Promise.resolve(null);
    if (!deps || typeof deps.digest !== "function") {
      return Promise.resolve({ ok: false, reason: "unverifiable", message: UNVERIFIABLE });
    }
    var name = "cropped.png";
    function fail(reason, message) { var e = new Error(reason); e.result = { ok: false, reason: reason, message: message }; return e; }
    return Promise.resolve()
      .then(function () {
        return deps.fetch(cropImageUrl(handover.sid, handover.crop), { credentials: "same-origin", cache: "no-store" });
      })
      .then(function (resp) {
        if (!resp || !resp.ok) throw fail("unavailable", UNAVAILABLE);
        name = fileName(resp.headers && resp.headers.get ? resp.headers.get("Content-Disposition") : "");
        return resp.arrayBuffer();
      })
      .then(function (buf) {
        if (!buf || buf.byteLength === 0 || buf.byteLength > MAX_BYTES) throw fail("unavailable", UNAVAILABLE);
        return Promise.resolve(deps.digest(buf)).then(function (digest) {
          if (hex(digest) !== handover.imageSha256) throw fail("mismatch", MISMATCH);
          var size = pngSize(new Uint8Array(buf));
          if (!size) throw fail("mismatch", MISMATCH);
          return { ok: true, bytes: buf, name: name, width: size.width, height: size.height };
        });
      })
      .catch(function (err) {
        return err && err.result ? err.result : { ok: false, reason: "unavailable", message: UNAVAILABLE };
      });
  }

  var api = { fetchCropped: fetchCropped, stripFragment: stripFragment, cropImageUrl: cropImageUrl,
              pngSize: pngSize, MESSAGES: { unavailable: UNAVAILABLE, mismatch: MISMATCH, unverifiable: UNVERIFIABLE } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TeachPreattach = api;
})(typeof window !== "undefined" ? window : this);
