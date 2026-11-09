// Test harness: runs the real teach/static/teach/{fragment,preattach,teach}.js in node against a
// minimal fake DOM, then performs scripted steps and prints JSON snapshots of the page state.
// Input (stdin JSON): {root, hash, settings: [{name, kind, default, min, max, options}],
//   crop: {url: {status, b64, disposition} | "reject"}, subtle: bool, steps: [...]}.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const elements = {};
  const replaceCalls = [];
  const fetchCalls = [];
  const events = [];
  let focused = null;

  function fake(id, props) {
    const e = Object.assign({
      id, hidden: false, textContent: "", value: "", className: "", files: [], checked: false,
      dataset: {}, listeners: {}, clicks: 0, src: "", tagName: "DIV",
      addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
      dispatch(t) { (this.listeners[t] || []).forEach((f) => f.call(this, { preventDefault() {} })); },
      click() { this.clicks += 1; this.dispatch("click"); },
      focus() { focused = this.id; },
      appendChild() {},
    }, props || {});
    elements[id] = e;
    return e;
  }

  // Initial visibility as rendered by teach/templates/teach/index.html.
  for (const id of ["preattach", "preattach-status", "mixup-warning", "settings-source"]) fake(id, { hidden: true });
  const settingInputs = input.settings.map((s) => {
    const id = "s-" + s.name;
    const base = { dataset: { setting: s.name, kind: s.kind } };
    if (s.kind === "boolean") return fake(id, Object.assign(base, { type: "checkbox", checked: s.default, defaultChecked: s.default }));
    if (s.kind === "string") {
      const options = s.options.map((v) => ({ value: v }));
      return fake(id, Object.assign(base, { tagName: "SELECT", value: s.default, options,
        querySelector: () => ({ value: s.default }) }));
    }
    return fake(id, Object.assign(base, { type: "number", min: String(s.min), max: String(s.max),
      value: String(s.default), defaultValue: String(s.default) }));
  });

  const location = { hash: input.hash || "", pathname: "/teach", search: "" };
  const history = {
    state: null,
    replaceState(state, title, url) { replaceCalls.push(url); events.push("replaceState " + url); location.hash = ""; },
  };

  const document = {
    getElementById(id) { return elements[id] || fake(id); },
    querySelectorAll(sel) { return sel === "[data-setting]" ? settingInputs : []; },
    querySelector(sel) { const m = /label\[for='s-(.+)'\]/.exec(sel); return { textContent: m ? m[1] : "" }; },
    createElement(tag) { return fake("_" + tag + Math.random(), { tagName: tag.toUpperCase() }); },
    createTextNode(t) { return { textContent: t }; },
  };

  class FakeFileReader {
    readAsDataURL(blob) {
      blob.arrayBuffer().then((b) => {
        this.result = "data:" + blob.type + ";base64," + Buffer.from(b).toString("base64");
        if (this.onload) this.onload();
      });
    }
  }

  let lastForm = null;
  async function fakeFetch(url, opts) {
    const call = { url, method: (opts && opts.method) || "GET" };
    fetchCalls.push(call);
    events.push("fetch " + call.method + " " + url);
    if (opts && opts.body instanceof FormData) {
      const entries = [];
      for (const [k, v] of opts.body.entries()) {
        if (typeof v === "string") entries.push({ key: k, text: v });
        else entries.push({ key: k, name: v.name, type: v.type, b64: Buffer.from(await v.arrayBuffer()).toString("base64") });
      }
      lastForm = entries;
      return { status: 200, ok: true, json: async () => ({ ok: true, errors: [], warnings: [], result: null }) };
    }
    const spec = (input.crop || {})[url];
    if (spec === "reject") throw new TypeError("network error");
    if (spec === undefined) return { ok: false, status: 404, headers: { get: () => null }, arrayBuffer: async () => new ArrayBuffer(0) };
    const bytes = Buffer.from(spec.b64, "base64");
    return {
      ok: spec.status >= 200 && spec.status < 300, status: spec.status,
      headers: { get: (h) => (h.toLowerCase() === "content-disposition" ? spec.disposition || null : null) },
      arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
    };
  }

  const win = {
    document, location, history, FormData, Blob, File, FileReader: FakeFileReader, TextEncoder, TextDecoder,
    atob, btoa, setTimeout, clearTimeout, Buffer,
    fetch: fakeFetch,
    crypto: input.subtle === false ? {} : { subtle: globalThis.crypto.subtle },
  };
  win.window = win;
  const ctx = vm.createContext(win);
  const dir = path.join(input.root, "teach", "static", "teach");
  for (const f of ["fragment.js", "preattach.js", "teach.js"]) {
    vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), ctx, { filename: f });
  }

  const settle = () => new Promise((r) => setTimeout(r, 50));
  function snapshot(label) {
    const g = (id) => elements[id] || {};
    return {
      label, hash: location.hash, replaceCalls: replaceCalls.slice(),
      fetchUrls: fetchCalls.map((c) => c.method + " " + c.url),
      pickHidden: !!g("image-pick").hidden, panelHidden: !!g("preattach").hidden,
      status: g("preattach-status").hidden ? null : g("preattach-status").textContent,
      statusClass: g("preattach-status").className || "",
      name: g("preattach-name").textContent || "", size: g("preattach-size").textContent || "",
      preview: g("preattach-preview").src || "", mixupHidden: !!g("mixup-warning").hidden,
      inputClicks: g("image").clicks || 0, focused, settingsFromApp: !g("settings-source").hidden,
      form: lastForm, events: events.slice(),
    };
  }

  const out = [];
  await settle();
  out.push(snapshot("load"));
  for (const step of input.steps || []) {
    if (step.do === "replace") document.getElementById("preattach-replace").click();
    else if (step.do === "pick") {
      const bytes = Buffer.from(step.b64, "base64");
      document.getElementById("image").files = [new File([bytes], step.name, { type: "image/png" })];
      document.getElementById("image").dispatch("change");
    } else if (step.do === "validate") {
      document.getElementById("title").value = step.title || "A title";
      document.getElementById("btn-validate").click();
    }
    await settle();
    out.push(snapshot(step.do));
  }
  process.stdout.write(JSON.stringify(out));
}

main().catch((e) => { process.stderr.write(String((e && e.stack) || e)); process.exit(1); });
