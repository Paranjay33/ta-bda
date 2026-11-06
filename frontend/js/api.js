/* Thin fetch wrapper: one place for the base URL, timeouts and the API's
   error contract ({ error: { code, message, request_id } }). */
(function () {
  "use strict";
  var cfg = window.HINGLISHSENSE_CONFIG || { apiBase: "/api/v1" };

  function ApiError(status, code, message, requestId) {
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.message = message;
    this.requestId = requestId;
  }
  ApiError.prototype = Object.create(Error.prototype);

  async function request(path, options) {
    options = options || {};
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, cfg.requestTimeoutMs || 60000);
    var headers = Object.assign({}, options.headers || {});
    if (cfg.apiKey) headers["X-API-Key"] = cfg.apiKey;
    if (options.json !== undefined) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
    }
    var res;
    try {
      res = await fetch((path.indexOf("http") === 0 ? "" : cfg.apiBase) + path, {
        method: options.method || "GET", headers: headers, body: options.body, signal: ctrl.signal
      });
    } catch (e) {
      throw new ApiError(0, e.name === "AbortError" ? "timeout" : "network_error",
        e.name === "AbortError" ? "The request timed out." : "Could not reach the HinglishSense API.", null);
    } finally {
      clearTimeout(timer);
    }
    var body = null;
    try { body = await res.json(); } catch (e) { body = null; }
    if (!res.ok) {
      var err = (body && body.error) || {};
      throw new ApiError(res.status, err.code || "http_" + res.status,
        err.message || ("Request failed (" + res.status + ")."), err.request_id || res.headers.get("X-Request-ID"));
    }
    return body;
  }

  window.HS_API = {
    ApiError: ApiError,
    get: function (p) { return request(p); },
    post: function (p, json) { return request(p, { method: "POST", json: json }); },
    upload: function (p, formData) { return request(p, { method: "POST", body: formData }); },
    del: function (p) { return request(p, { method: "DELETE" }); },
    url: function (p) { return cfg.apiBase + p; }
  };
})();
