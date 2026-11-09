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
  }
  async function load() {
    await environment();
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
