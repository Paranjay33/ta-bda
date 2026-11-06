/* ==========================================================================
   HinglishSense — research dashboard, batch analysis, history & analytics.
   Every number is read from the API (saved artifacts / database). Empty
   states are shown instead of placeholder charts.
   ========================================================================== */
(function () {
  "use strict";
  var API = window.HS_API, UI = window.HS_UI;
  var el = UI.el, pct = UI.pct, cap = UI.cap;
  var ORDER = ["negative", "neutral", "positive"];
  function $(id) { return document.getElementById(id); }

  function table(target, headers, rows, emptyText) {
    var t = $(target);
    t.replaceChildren();
    if (!rows.length) {
      t.appendChild(el("tbody", {}, [el("tr", {}, [el("td", { class: "data-table__empty", text: emptyText || "No data yet." })])]));
      return;
    }
    t.appendChild(el("thead", {}, [el("tr", {}, headers.map(function (h) { return el("th", { scope: "col", text: h }); }))]));
    t.appendChild(el("tbody", {}, rows.map(function (r) {
      return el("tr", {}, r.map(function (c) { return c instanceof Node ? el("td", {}, [c]) : el("td", { text: c === null || c === undefined ? "—" : String(c) }); }));
    })));
  }

  function distBars(target, counts, kind) {
    var box = $(target);
    box.replaceChildren();
    var keys = Object.keys(counts);
    var total = keys.reduce(function (a, k) { return a + counts[k]; }, 0);
    if (!total) { box.appendChild(el("p", { class: "panel__hint", text: "No records yet." })); return; }
    keys.forEach(function (k) {
      var v = counts[k];
      var fill = el("div", { class: "prob-row__fill " + (kind === "sentiment" ? "prob-row__fill--" + k : "prob-row__fill--primary") });
      fill.style.width = (100 * v / total).toFixed(1) + "%";
      box.appendChild(el("div", { class: "prob-row prob-row--wide" }, [
        el("span", { class: "prob-row__label", text: kind === "sentiment" ? cap(k) : k.toUpperCase() }),
        el("div", { class: "prob-row__bar" }, [fill]),
        el("span", { class: "prob-row__value", text: v + " (" + (100 * v / total).toFixed(0) + "%)" })
      ]));
    });
  }

  /* --------------------------------------------------------------- research */
  async function loadResearch() {
    try {
      var d = await API.get("/metrics/comparison");
      var s = d.split || {};
      var rows = s.rows || {};
      $("researchSplit").textContent = "Frozen split (" + (s.policy || "—") + "): train " + rows.train + " · validation " +
        rows.validation + " · test " + rows.test + " unique tweets after removing " +
        ((s.counts || {}).collapsed_exact_duplicates || 0) + " duplicates and " +
        ((s.counts || {}).dropped_conflicting_label_rows || 0) + " conflicting-label rows. " +
        "\"not run\" means no artifact exists — it is never a zero.";
      table("metricsTable",
        ["Exp", "Model", "Split", "Accuracy", "Macro-F1", "Weighted-F1", "F1 neg", "F1 neu", "F1 pos", "Val Macro-F1", "Log-loss", "ECE"],
        d.rows.map(function (r) {
          return [r.experiment, r.model, r.evaluation_split, pct(r.test_accuracy, 2), pct(r.test_macro_f1, 2), pct(r.test_weighted_f1, 2),
            pct(r.test_f1_negative), pct(r.test_f1_neutral), pct(r.test_f1_positive), pct(r.validation_macro_f1, 2),
            r.test_log_loss === null ? "not run" : r.test_log_loss.toFixed(3), r.test_ece === null ? "not run" : r.test_ece.toFixed(3)];
        }));
      var sig = d.e3_e4_significance;
      $("significanceText").textContent = sig
        ? "E4 − E3 Macro-F1 = " + (100 * sig.macro_f1_diff_e4_minus_e3).toFixed(2) + " pp; paired bootstrap 95% CI [" +
          (100 * sig.ci_low).toFixed(2) + ", " + (100 * sig.ci_high).toFixed(2) + "] pp (" + sig.n_bootstrap +
          " resamples); McNemar exact p = " + Number(sig.mcnemar_p).toPrecision(3) + "."
        : "E3/E4 significance test not available until both have a locked test evaluation.";

      var grid = $("cmGrid");
      grid.replaceChildren();
      Object.keys(d.confusion_matrices).forEach(function (exp) {
        var cm = d.confusion_matrices[exp];
        var card = el("div", { class: "cm-card" }, [el("p", { class: "panel__title", text: exp })]);
        if (!cm) { card.appendChild(el("p", { class: "panel__hint", text: "not run" })); grid.appendChild(card); return; }
        var max = Math.max.apply(null, cm.map(function (r) { return Math.max.apply(null, r); })) || 1;
        var t = el("table", { class: "cm" });
        t.appendChild(el("tr", {}, [el("th", { text: "" })].concat(d.labels_order.map(function (l) { return el("th", { scope: "col", text: l.slice(0, 3) }); }))));
        cm.forEach(function (row, i) {
          t.appendChild(el("tr", {}, [el("th", { scope: "row", text: d.labels_order[i].slice(0, 3) })].concat(row.map(function (v, j) {
            var td = el("td", { text: String(v) });
            td.style.setProperty("--a", (v / max).toFixed(3));
            if (i === j) td.className = "cm__diag";
            return td;
          }))));
        });
        card.appendChild(t);
        grid.appendChild(card);
      });

      var ea = d.error_analysis || {};
      var hints = ["hint_negation", "hint_contrast", "hint_emoji", "hint_code_mixing", "hint_short_text"];
      table("errorTable", ["Exp", "Error rate"].concat(hints.map(function (h) { return h.replace("hint_", "").replace("_", " ") + " (n / err)"; })),
        Object.keys(ea).filter(function (k) { return typeof ea[k] === "object"; }).map(function (k) {
          var s = ea[k];
          return [k, pct(s.error_rate)].concat(hints.map(function (h) {
            var v = s.by_hint[h];
            return v.n + " / " + (v.error_rate === null ? "—" : pct(v.error_rate));
          }));
        }), "Error analysis appears after the locked test evaluation has been run.");

      // Research insight (top of page)
      var e3 = d.rows.find(function (r) { return r.experiment === "E3"; });
      var e4 = d.rows.find(function (r) { return r.experiment === "E4"; });
      if (e3 && e4 && e3.test_macro_f1 !== null && e4.test_macro_f1 !== null) {
        var diff = 100 * (e4.test_macro_f1 - e3.test_macro_f1);
        $("insightStat").textContent = (diff >= 0 ? "+" : "") + diff.toFixed(2) + " pp Macro-F1";
        $("insightBody").textContent = "HingRoBERTa (E4) " + pct(e4.test_macro_f1, 2) + " vs XLM-R (E3) " + pct(e3.test_macro_f1, 2) +
          " Macro-F1 on " + e4.test_rows + " held-out tweets, trained under one matched protocol where only the pretrained checkpoint differs." +
          (sig ? " 95% bootstrap CI: [" + (100 * sig.ci_low).toFixed(2) + ", " + (100 * sig.ci_high).toFixed(2) + "] pp." : "") +
          " One dataset and one seed — not a guarantee for new text.";
      } else {
        $("insightStat").textContent = "Not yet measured";
        $("insightBody").textContent = "The verified E3/E4 comparison appears here once both models have a locked test evaluation.";
      }
    } catch (e) {
      $("researchSplit").textContent = "Could not load evaluation artifacts (" + e.message + ").";
      $("insightStat").textContent = "—";
      $("insightBody").textContent = "Research results are unavailable because the API could not be reached.";
    }
  }

  /* --------------------------------------------------------------- batch */
  var pollTimer = null;
  function renderJob(job) {
    $("batchResult").hidden = false;
    $("batchStatus").textContent = "Job " + job.id.slice(0, 8) + " — " + job.status;
    $("batchMeta").textContent = (job.input_filename || "JSON batch") + " · column \"" + (job.text_column || "-") + "\" · model " +
      (job.model_requested === "auto" ? "Auto → " : "") + (job.model_used || "?").toUpperCase() + " (" + (job.model_version || "—") + ") · " +
      job.processed_count + "/" + job.row_count + " processed, " + job.failed_count + " skipped" +
      (job.summary && job.summary.mean_inference_ms_per_row ? " · " + job.summary.mean_inference_ms_per_row + " ms/row" : "") +
      (job.error ? " · " + job.error : "");
    var running = job.status === "queued" || job.status === "running";
    $("batchProgressWrap").hidden = !running;
    $("batchProgress").style.width = (job.row_count ? 100 * job.processed_count / job.row_count : 0).toFixed(0) + "%";
    var dl = $("batchDownload");
    dl.hidden = !job.download_url;
    if (job.download_url) dl.href = job.download_url;
    if (job.summary && job.summary.distribution) distBars("batchDist", job.summary.distribution, "sentiment");
    else $("batchDist").replaceChildren();
    var rows = job.rows || [];
    table("batchTable", ["Row", "Text", "Status", "Prediction", "Negative", "Neutral", "Positive"],
      rows.slice(0, 50).map(function (r) {
        return [r.row, r.text.length > 90 ? r.text.slice(0, 90) + "…" : r.text, r.status === "ok" ? "ok" : "skipped: " + r.error,
          r.prediction ? cap(r.prediction) : "—", r.score_negative ? pct(+r.score_negative) : "—",
          r.score_neutral ? pct(+r.score_neutral) : "—", r.score_positive ? pct(+r.score_positive) : "—"];
      }), running ? "Processing…" : "No rows.");
    if (rows.length > 50) $("batchTable").appendChild(el("caption", { class: "panel__hint", text: "Showing the first 50 rows; download the CSV for all of them." }));
    if (running) { clearTimeout(pollTimer); pollTimer = setTimeout(function () { pollJob(job.id); }, 1500); }
    else document.dispatchEvent(new CustomEvent("hs:analysis-done"));
  }
  async function pollJob(id) {
    try { renderJob(await API.get("/batch/jobs/" + id)); }
    catch (e) { $("batchStatus").textContent = "Could not fetch job status: " + e.message; }
  }
  $("batchForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    var err = $("batchError");
    err.hidden = true;
    var f = $("batchFile").files[0];
    if (!f) { err.textContent = "Choose a CSV file first."; err.hidden = false; return; }
    var fd = new FormData();
    fd.append("file", f);
    fd.append("model", $("batchModel").value);
    if ($("batchColumn").value.trim()) fd.append("text_column", $("batchColumn").value.trim());
    var btn = $("batchSubmit");
    btn.disabled = true; btn.textContent = "Uploading…";
    try { renderJob(await API.upload("/predict/batch", fd)); }
    catch (ex) { err.textContent = ex.message; err.hidden = false; }
    finally { btn.disabled = false; btn.textContent = "Run batch"; }
  });

  /* --------------------------------------------------------------- history & analytics */
  var page = 1, pageSize = 10;
  async function loadAnalytics() {
    try {
      var a = await API.get("/analytics/summary");
      var stats = $("analyticsStats");
      stats.replaceChildren(
        el("div", { class: "stat" }, [el("span", { class: "stat__value", text: String(a.total) }), el("span", { class: "stat__label", text: "Stored predictions" })]),
        el("div", { class: "stat" }, [el("span", { class: "stat__value", text: String(a.by_source.single || 0) }), el("span", { class: "stat__label", text: "Single" })]),
        el("div", { class: "stat" }, [el("span", { class: "stat__value", text: String(a.by_source.compare || 0) }), el("span", { class: "stat__label", text: "Compare" })]),
        el("div", { class: "stat" }, [el("span", { class: "stat__value", text: String(a.by_source.batch || 0) }), el("span", { class: "stat__label", text: "Batch rows" })])
      );
      distBars("analyticsDist", a.by_prediction, "sentiment");
      distBars("analyticsModels", a.by_model, "model");
      distBars("analyticsDays", a.per_day, "day");
      table("analyticsLatency", ["Model", "Requests", "Mean", "Median"], Object.keys(a.latency_single_requests).map(function (m) {
        var l = a.latency_single_requests[m];
        return [m.toUpperCase(), l.n, l.mean_ms + " ms", l.median_ms + " ms"];
      }), "No single requests yet.");
    } catch (e) {
      $("analyticsStats").replaceChildren(el("p", { class: "panel__hint", text: "Analytics unavailable: " + e.message }));
    }
  }
  async function loadHistory() {
    var q = "?page=" + page + "&page_size=" + pageSize;
    if ($("histModel").value) q += "&model=" + $("histModel").value;
    if ($("histPred").value) q += "&prediction=" + $("histPred").value;
    if ($("histSource").value) q += "&source=" + $("histSource").value;
    try {
      var h = await API.get("/analyses" + q);
      $("historyPrivacy").textContent = h.text_storage_enabled
        ? "Text storage is ON for this deployment: the normalized text of each request is kept in the local database."
        : "Privacy: the text you analyze is not stored (ALLOW_HISTORY_STORAGE=false). Only the prediction, scores, model and timing are kept.";
      table("historyTable", ["When", "Source", "Model", "Version", "Prediction", "Scores (neg / neu / pos)", "Time"].concat(h.text_storage_enabled ? ["Normalized text"] : []),
        h.items.map(function (r) {
          var row = [new Date(r.created_at).toLocaleString(), r.source, (r.model_requested === "auto" ? "Auto → " : "") + r.model_used.toUpperCase(),
            r.model_version, cap(r.prediction), ORDER.map(function (k) { return pct(r.scores[k], 0); }).join(" / "), Math.round(r.inference_ms) + " ms"];
          if (h.text_storage_enabled) row.push(r.normalized_text || "—");
          return row;
        }), "No analyses stored yet.");
      var pages = Math.max(1, Math.ceil(h.total / pageSize));
      $("histPage").textContent = "Page " + page + " of " + pages + " · " + h.total + " records";
      $("histPrev").disabled = page <= 1;
      $("histNext").disabled = page >= pages;
    } catch (e) {
      table("historyTable", [], [], "History unavailable: " + e.message);
    }
  }
  ["histModel", "histPred", "histSource"].forEach(function (id) { $(id).addEventListener("change", function () { page = 1; loadHistory(); }); });
  $("histRefresh").addEventListener("click", function () { loadHistory(); loadAnalytics(); });
  $("histPrev").addEventListener("click", function () { if (page > 1) { page--; loadHistory(); } });
  $("histNext").addEventListener("click", function () { page++; loadHistory(); });
  $("histClear").addEventListener("click", async function () {
    if (!window.confirm("Delete all stored analysis history in this deployment?")) return;
    try { await API.del("/analyses"); page = 1; loadHistory(); loadAnalytics(); }
    catch (e) { window.alert("Could not clear history: " + e.message); }
  });

  document.addEventListener("hs:analysis-done", function () { loadHistory(); loadAnalytics(); });
  document.addEventListener("hs:registry", function () {
    var lim = UI.registry.limits;
    $("batchLimits").textContent = "Up to " + lim.max_batch_rows + " rows · " + lim.max_upload_mb + " MB · " + lim.max_text_chars + " characters per row";
  });

  loadResearch();
  loadHistory();
  loadAnalytics();
})();
