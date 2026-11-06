/* HinglishSense runtime configuration.
   Same-origin deployment (the FastAPI server also serves this frontend) needs
   no change. To point a separately hosted frontend at another API, set apiBase
   to an absolute URL (and add this origin to the API's CORS_ORIGINS). */
window.HINGLISHSENSE_CONFIG = window.HINGLISHSENSE_CONFIG || {
  apiBase: "/api/v1",
  apiKey: null,            // only when the API runs with AUTH_REQUIRED=true
  requestTimeoutMs: 60000
};
