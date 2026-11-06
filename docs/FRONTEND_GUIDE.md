# Frontend guide

The supplied HinglishSense frontend (`hinglishsense.zip`) is the product UI. It was extracted to
`frontend/` and extended in place — same HTML structure, CSS tokens (cream `#FAF8F3`, purple
`#473C8C`, Newsreader + Inter), components and accessible listbox. No framework, no build step.

| File | Role |
|---|---|
| `index.html` | supplied page + new sections (Compare, Batch, Research, History & analytics) |
| `css/style.css` | supplied styles + an appended block built only from the existing tokens |
| `js/config.js` | `apiBase` (default `/api/v1`, same origin), optional `apiKey`, timeout |
| `js/api.js` | fetch wrapper: base URL, timeout, error contract → `ApiError` |
| `js/app.js` | analyzer: input, chips, listbox, live model status, predict, compare, result states |
| `js/dashboard.js` | research dashboard, insight, batch flow, history, analytics |

## Behaviour

- Model selector options are populated from `GET /api/v1/models`: unavailable models are
  `aria-disabled`, skipped by Arrow keys, and show the reason. Auto shows what it resolves to.
  Keyboard: Arrow Up/Down, Home/End, Enter/Space select, Escape closes, Tab closes.
- Analyze: validates (non-empty, ≤ limit from `/status`), disables buttons (`aria-busy`),
  shows a spinner, renders the real response, announces the result via an `aria-live` region,
  scrolls (respecting `prefers-reduced-motion`). Ctrl/⌘+Enter submits.
- Result: sentiment badge (text + colour), "Calibrated probability: x%" or "Top class score: x%
  (uncalibrated)", model actually used (`Auto → …`), version, inference ms, score-type
  explanation, bars in fixed order, collapsible "What the model actually read", warnings.
- States: waiting · loading · done · error (input / unavailable / rate-limited / unreachable).
- Compare E3 vs E4: two cards side by side, independent status, version, latency, score type,
  agreement line; no "winner" from a single sentence.
- Examples: tags are hand-written illustrations; clicking only fills the box.
- Models / insight: values come from the API (locked test → else validation → else "not
  evaluated"). The midterm numbers appear only in a labelled "not reproducible" box.
- Batch: CSV upload, optional column, model; progress for background jobs; distribution,
  first 50 rows, download link.
- Research: metrics table (`not run` is never shown as 0), significance, confusion matrices,
  heuristic error hints.
- History & analytics: privacy notice, stat tiles, CSS bar charts, latency table, filters,
  pagination, clear. Empty states instead of placeholder charts.

All user-provided text is inserted with `textContent` (never `innerHTML`).

## Running without the API server

Serve `frontend/` from any static server and set `window.HINGLISHSENSE_CONFIG.apiBase` in
`js/config.js` to the API URL; add the UI origin to the API's `CORS_ORIGINS`.
