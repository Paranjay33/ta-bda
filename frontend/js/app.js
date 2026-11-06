/* ==========================================================================
   HinglishSense — analyzer interactivity
   Vanilla JS. Every prediction shown comes from the API; nothing is invented.
   Model availability, versions and metrics come from GET /api/v1/models.
   ========================================================================== */

(function () {
  "use strict";

  var API = window.HS_API;
  var LABELS = ["positive", "neutral", "negative"]; // display order (bars)
  var SCORE_TYPE_TEXT = {
    calibrated_probability: "Calibrated probability — calibration fitted on the validation split only.",
    model_probability: "Model probability — not calibrated (known to be over-confident); read as a relative score."
  };

  /* ---------------------------------------------------------------------
     Static model copy (display text only — availability is from the API)
     --------------------------------------------------------------------- */
  var MODELS = {
    auto: { display: "Auto", desc: "Backend picks the preferred validated model" },
    e1: { display: "E1 — TF-IDF + Logistic Regression", desc: "Word-frequency features with a linear classifier — fast, interpretable baseline (Spark MLlib)" },
    e2: { display: "E2 — Word + Character Features + SVM", desc: "Adds character n-grams to catch spelling variation across Hinglish (Spark MLlib)" },
    e3: { display: "E3 — XLM-R", desc: "General-purpose multilingual transformer, fine-tuned on Hinglish sentiment" },
    e4: { display: "E4 — HingRoBERTa", desc: "Transformer pre-trained specifically on Hindi-English code-mixed text" }
  };
  var REASONS = {
    not_trained: "not trained yet",
    training_incomplete: "training incomplete",
    artifact_missing: "artifact missing",
    calibration_missing: "calibration missing",
    spark_service_unreachable: "Spark inference service not running",
    spark_model_not_loaded: "not loaded in the Spark service",
    spark_service_version_mismatch: "Spark service has a different model version"
  };
  var registry = { models: {}, auto: null, limits: { max_text_chars: 500 } };
  var selectedModel = "auto";
  var busy = false;

  /* ---------------------------------------------------------------------
     Elements
     --------------------------------------------------------------------- */
  function $(id) { return document.getElementById(id); }
  var navToggle = $("navToggle"), navMobile = $("navMobile");
  var textarea = $("inputText"), clearBtn = $("clearBtn"), charCount = $("charCount");
  var chips = document.querySelectorAll(".chip");
  var modelTrigger = $("modelTrigger"), modelList = $("modelList");
  var modelTriggerName = $("modelTriggerName"), modelTriggerDesc = $("modelTriggerDesc");
  var modelOptions = Array.from(modelList.querySelectorAll(".model-select__option"));
  var analyzeBtn = $("analyzeBtn"), compareBtn = $("compareBtn"), inputError = $("inputError");

  var states = { waiting: $("stateWaiting"), loading: $("stateLoading"), error: $("stateError"), done: $("stateDone") };
  var announce = $("resultAnnounce");

  /* ---------------------------------------------------------------------
     Small helpers (also used by dashboard.js)
     --------------------------------------------------------------------- */
  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") n.textContent = attrs[k];
      else if (k === "class") n.className = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function pct(v, digits) { return (v === null || v === undefined) ? "not run" : (100 * v).toFixed(digits === undefined ? 1 : digits) + "%"; }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  window.HS_UI = { el: el, pct: pct, cap: cap, registry: registry, reasons: REASONS };

  /* ---------------------------------------------------------------------
     Mobile nav (unchanged behaviour)
     --------------------------------------------------------------------- */
  navToggle.addEventListener("click", function () {
    var open = navMobile.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  });
  navMobile.querySelectorAll("a").forEach(function (link) {
    link.addEventListener("click", function () {
      navMobile.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
      navToggle.setAttribute("aria-label", "Open menu");
    });
  });

  /* ---------------------------------------------------------------------
     Textarea: auto-grow + character counter
     --------------------------------------------------------------------- */
  function maxChars() { return registry.limits.max_text_chars || 500; }
  function autoGrow() {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 320) + "px";
  }
  function updateCharCount() {
    var len = textarea.value.length;
    charCount.textContent = len + "/" + maxChars();
    charCount.classList.toggle("char-count--near-limit", len > maxChars() * 0.9);
  }
  textarea.addEventListener("input", function () { autoGrow(); updateCharCount(); hideInputError(); });
  textarea.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runAnalyze(); }
  });
  clearBtn.addEventListener("click", function () {
    textarea.value = "";
    autoGrow(); updateCharCount(); hideInputError();
    textarea.focus();
  });

  /* Example chips — populate textarea, do NOT auto-run analysis */
  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      textarea.value = chip.getAttribute("data-text");
      autoGrow(); updateCharCount(); hideInputError();
      textarea.focus();
    });
  });

  /* ---------------------------------------------------------------------
     Custom accessible model selector (ARIA listbox pattern)
     --------------------------------------------------------------------- */
  var activeIndex = 0;
  function isDisabled(o) { return o.getAttribute("aria-disabled") === "true"; }

  function openList() {
    modelList.hidden = false;
    modelTrigger.setAttribute("aria-expanded", "true");
    setActiveOption(modelOptions.findIndex(function (o) { return o.dataset.value === selectedModel; }));
    document.addEventListener("click", onOutsideClick);
  }
  function closeList() {
    modelList.hidden = true;
    modelTrigger.setAttribute("aria-expanded", "false");
    modelTrigger.removeAttribute("aria-activedescendant");
    document.removeEventListener("click", onOutsideClick);
  }
  function onOutsideClick(e) { if (!modelTrigger.parentElement.contains(e.target)) closeList(); }
  function setActiveOption(index) {
    if (index < 0) index = 0;
    if (index > modelOptions.length - 1) index = modelOptions.length - 1;
    activeIndex = index;
    modelOptions.forEach(function (o, i) { o.classList.toggle("is-active", i === index); });
    modelTrigger.setAttribute("aria-activedescendant", modelOptions[index].id);
    modelOptions[index].scrollIntoView({ block: "nearest" });
  }
  function moveActive(step) {
    var i = activeIndex;
    for (var n = 0; n < modelOptions.length; n++) {
      i += step;
      if (i < 0 || i >= modelOptions.length) return;
      if (!isDisabled(modelOptions[i])) { setActiveOption(i); return; }
    }
  }
  function triggerTextFor(value) {
    if (value === "auto") {
      var to = registry.auto && registry.auto.resolves_to;
      return { name: to ? "Auto → " + MODELS[to].display : "Auto (HingRoBERTa)",
               desc: to ? "Backend-selected validated model" : "No validated model is available yet" };
    }
    return { name: MODELS[value].display, desc: MODELS[value].desc };
  }
  function selectModel(value, keepFocus) {
    var opt = modelOptions.find(function (o) { return o.dataset.value === value; });
    if (!opt || isDisabled(opt)) return;
    selectedModel = value;
    var t = triggerTextFor(value);
    modelTriggerName.textContent = t.name;
    modelTriggerDesc.textContent = t.desc;
    modelOptions.forEach(function (o) { o.setAttribute("aria-selected", String(o.dataset.value === value)); });
    closeList();
    if (!keepFocus) modelTrigger.focus();
  }
  modelTrigger.addEventListener("click", function () {
    if (modelTrigger.getAttribute("aria-expanded") === "true") closeList(); else openList();
  });
  modelTrigger.addEventListener("keydown", function (e) {
    var isOpen = modelTrigger.getAttribute("aria-expanded") === "true";
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!isOpen) { openList(); return; }
      moveActive(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === "Home" && isOpen) { e.preventDefault(); setActiveOption(0);
    } else if (e.key === "End" && isOpen) { e.preventDefault(); setActiveOption(modelOptions.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (isOpen) selectModel(modelOptions[activeIndex].dataset.value); else openList();
    } else if (e.key === "Escape") {
      if (isOpen) { e.preventDefault(); closeList(); }
    } else if (e.key === "Tab" && isOpen) {
      closeList();
    }
  });
  modelOptions.forEach(function (option, i) {
    option.addEventListener("click", function (e) {
      if (isDisabled(option)) { e.stopPropagation(); return; }
      selectModel(option.dataset.value);
    });
    option.addEventListener("mouseenter", function () { setActiveOption(i); });
  });

  /* ---------------------------------------------------------------------
     Registry: model availability + Models section, from the API
     --------------------------------------------------------------------- */
  function statusText(m) {
    if (!m) return "Status unknown";
    if (!m.available) return "Unavailable — " + (REASONS[m.unavailable_reason] || m.unavailable_reason || "unknown");
    var bits = ["Ready"];
    if (m.test_metrics && m.test_metrics.macro_f1 !== null) bits.push("test Macro-F1 " + pct(m.test_metrics.macro_f1));
    else if (m.validation_macro_f1 !== null && m.validation_macro_f1 !== undefined) bits.push("val Macro-F1 " + pct(m.validation_macro_f1));
    return bits.join(" · ");
  }

  async function loadRegistry() {
    try {
      var data = await API.get("/models");
      var status = await API.get("/status").catch(function () { return null; });
      registry.auto = data.auto;
      if (status && status.limits) registry.limits = status.limits;
      textarea.setAttribute("maxlength", String(maxChars()));
      updateCharCount();
      data.models.forEach(function (m) { registry.models[m.model_id] = m; });

      modelOptions.forEach(function (o) {
        var v = o.dataset.value;
        var line = o.querySelector(".model-select__status");
        var ok, text;
        if (v === "auto") {
          ok = !!data.auto.resolves_to;
          text = ok ? "Resolves to " + data.auto.resolves_to.toUpperCase() + (data.auto.warnings.length ? " (fallback)" : "")
                    : "Unavailable — no validated model is loaded";
          o.querySelector(".model-select__option-name").textContent =
            ok ? "Auto (" + MODELS[data.auto.resolves_to].display.split("— ")[1] + ")" : "Auto";
        } else {
          ok = registry.models[v] && registry.models[v].available;
          text = statusText(registry.models[v]);
        }
        o.setAttribute("aria-disabled", String(!ok));
        o.classList.toggle("is-disabled", !ok);
        if (line) { line.textContent = text; line.classList.toggle("is-ok", !!ok); }
      });
      var current = modelOptions.find(function (o) { return o.dataset.value === selectedModel; });
      if (current && isDisabled(current)) {
        var first = modelOptions.find(function (o) { return !isDisabled(o); });
        if (first) selectModel(first.dataset.value, true);
      } else {
        selectModel(selectedModel, true);
      }
      compareBtn.disabled = !(registry.models.e3 && registry.models.e3.available) && !(registry.models.e4 && registry.models.e4.available);

      ["e1", "e2", "e3", "e4"].forEach(function (id) {
        var m = registry.models[id];
        var st = document.querySelector('[data-card-status="' + id + '"]');
        var metric = document.querySelector('[data-card-metric="' + id + '"]');
        if (st) { st.textContent = statusText(m); st.classList.toggle("is-ok", !!(m && m.available)); }
        if (metric) {
          var val = metric.querySelector(".model-card__metric-value");
          var lab = metric.querySelector(".model-card__metric-label");
          if (m && m.test_metrics) { val.textContent = pct(m.test_metrics.macro_f1, 2); lab.textContent = "Macro-F1 · locked test (" + m.test_split_rows + " tweets)"; }
          else if (m && m.validation_macro_f1 !== null && m.validation_macro_f1 !== undefined) { val.textContent = pct(m.validation_macro_f1, 2); lab.textContent = "Macro-F1 · validation (test not run)"; }
          else { val.textContent = "—"; lab.textContent = "Not evaluated yet"; }
        }
      });
      document.dispatchEvent(new CustomEvent("hs:registry", { detail: data }));
    } catch (e) {
      modelOptions.forEach(function (o) {
        var line = o.querySelector(".model-select__status");
        if (line) line.textContent = "API unreachable — status unknown";
      });
      modelTriggerDesc.textContent = "Cannot reach the HinglishSense API";
      document.querySelectorAll(".model-card__status").forEach(function (s) { s.textContent = "API unreachable"; });
    }
  }

  /* ---------------------------------------------------------------------
     Result rendering
     --------------------------------------------------------------------- */
  function show(name) {
    Object.keys(states).forEach(function (k) { states[k].hidden = k !== name; });
  }
  function showInputError(msg) { inputError.textContent = msg; inputError.hidden = false; }
  function hideInputError() { inputError.hidden = true; inputError.textContent = ""; }
  function setBusy(on, label) {
    busy = on;
    analyzeBtn.disabled = on; compareBtn.disabled = on;
    analyzeBtn.setAttribute("aria-busy", String(on));
    analyzeBtn.querySelector(".btn__label").textContent = on ? (label || "Analyzing…") : "Analyze Sentiment";
    if (!on) loadRegistry();
  }

  function renderPrediction(r) {
    show("done");
    var badge = $("sentimentBadge");
    badge.textContent = cap(r.prediction);
    badge.className = "result__sentiment-badge is-" + r.prediction;
    $("confidenceText").textContent = r.confidence !== null
      ? "Calibrated probability: " + pct(r.confidence)
      : "Top class score: " + pct(r.top_class_score) + " (uncalibrated)";
    var usedText = r.model_name;
    if (r.model_requested === "auto") usedText = "Auto → " + r.model_name;
    $("modelUsed").textContent = usedText;
    $("modelVersion").textContent = r.model_version || "—";
    $("processingTime").textContent = Math.round(r.inference_ms) + " ms";
    $("scoreTypeLabel").textContent = SCORE_TYPE_TEXT[r.score_type] || r.score_type;
    LABELS.forEach(function (lab) {
      var v = r.scores[lab];
      var key = cap(lab);
      $("bar" + key).style.width = (100 * v).toFixed(1) + "%";
      $("val" + key).textContent = pct(v);
    });
    $("modelInput").textContent = r.model_input || "(empty after preprocessing)";
    var notes = r.warnings && r.warnings.length ? r.warnings.join(" ") : "";
    $("resultNotice").textContent = notes || "Live prediction from the registered model artifact. A research prototype — not a judgement of the writer.";
    $("resultNotice").classList.toggle("is-warning", !!notes);
    announce.textContent = "Result: " + cap(r.prediction) + " according to " + r.model_name + ".";
  }

  function renderError(err, unavailableTitle) {
    show("error");
    var title = "Something went wrong";
    if (err.status === 503) title = unavailableTitle || "Model unavailable";
    else if (err.status === 400 || err.status === 422) title = "Please check your input";
    else if (err.status === 429) title = "Too many requests";
    else if (err.status === 0) title = "API unreachable";
    $("errorTitle").textContent = title;
    $("errorBody").textContent = err.message + (err.requestId ? " (request " + err.requestId.slice(0, 8) + ")" : "");
    announce.textContent = title + ". " + err.message;
  }

  function validateInput() {
    var text = textarea.value.trim();
    if (!text) { showInputError("Type or paste a sentence first."); textarea.focus(); return null; }
    if (text.length > maxChars()) { showInputError("Please keep it under " + maxChars() + " characters."); return null; }
    return text;
  }

  function scrollToResult(id) {
    var target = $(id);
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }

  async function analyzeSentiment(text, model) {
    return API.post("/predict", { text: text, model: model });
  }

  async function runAnalyze() {
    if (busy) return;
    var text = validateInput();
    if (!text) return;
    hideInputError();
    $("compare").hidden = true;
    setBusy(true);
    show("loading");
    $("loadingText").textContent = "Running " + (selectedModel === "auto" ? "the Auto-selected model" : MODELS[selectedModel].display) + "…";
    scrollToResult("result");
    try {
      renderPrediction(await analyzeSentiment(text, selectedModel));
      document.dispatchEvent(new CustomEvent("hs:analysis-done"));
    } catch (err) {
      renderError(err);
    } finally {
      setBusy(false);
    }
  }

  /* ---------------------------------------------------------------------
     E3 vs E4 comparison
     --------------------------------------------------------------------- */
  function compareCard(id, r) {
    var card = el("article", { class: "compare-card" + (r.status === "ok" ? "" : " compare-card--off") });
    card.appendChild(el("p", { class: "compare-card__id", text: id.toUpperCase() }));
    card.appendChild(el("p", { class: "compare-card__name", text: r.model_name }));
    if (r.status !== "ok") {
      card.appendChild(el("p", { class: "compare-card__unavailable", text: "Unavailable — " + (REASONS[r.reason] || r.reason) }));
      return card;
    }
    card.appendChild(el("p", { class: "result__sentiment-badge is-" + r.prediction, text: cap(r.prediction) }));
    card.appendChild(el("p", { class: "result__confidence", text: r.confidence !== null ? "Calibrated probability " + pct(r.confidence) : "Top class score " + pct(r.top_class_score) }));
    var probs = el("div", { class: "result__probs" });
    LABELS.forEach(function (lab) {
      var fill = el("div", { class: "prob-row__fill prob-row__fill--" + lab });
      fill.style.width = (100 * r.scores[lab]).toFixed(1) + "%";
      probs.appendChild(el("div", { class: "prob-row" }, [
        el("span", { class: "prob-row__label", text: cap(lab) }),
        el("div", { class: "prob-row__bar" }, [fill]),
        el("span", { class: "prob-row__value", text: pct(r.scores[lab]) })
      ]));
    });
    card.appendChild(probs);
    card.appendChild(el("p", { class: "compare-card__meta", text: (r.model_version || "—") + " · " + Math.round(r.inference_ms) + " ms" }));
    card.appendChild(el("p", { class: "compare-card__meta", text: SCORE_TYPE_TEXT[r.score_type] || r.score_type }));
    return card;
  }

  async function runCompare() {
    if (busy) return;
    var text = validateInput();
    if (!text) return;
    hideInputError();
    setBusy(true, "Comparing…");
    var section = $("compare"), grid = $("compareGrid"), agree = $("compareAgreement");
    section.hidden = false;
    grid.replaceChildren(el("p", { class: "panel__hint", text: "Running E3 and E4 on the same normalized input…" }));
    agree.textContent = "";
    scrollToResult("compare");
    try {
      var data = await API.post("/predict/compare", { text: text, models: ["e3", "e4"] });
      grid.replaceChildren(compareCard("e3", data.results.e3), compareCard("e4", data.results.e4));
      if (data.agreement === true) agree.textContent = "Both models agree on this sentence.";
      else if (data.agreement === false) agree.textContent = "The models disagree on this sentence.";
      else agree.textContent = "Only one model could run, so agreement cannot be assessed.";
      document.dispatchEvent(new CustomEvent("hs:analysis-done"));
    } catch (err) {
      grid.replaceChildren(el("p", { class: "result__error-title", text: err.message }));
    } finally {
      setBusy(false);
    }
  }

  analyzeBtn.addEventListener("click", runAnalyze);
  compareBtn.addEventListener("click", runCompare);

  /* ---------------------------------------------------------------------
     Init
     --------------------------------------------------------------------- */
  autoGrow();
  updateCharCount();
  show("waiting");
  loadRegistry();
})();
