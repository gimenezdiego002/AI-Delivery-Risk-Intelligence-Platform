# Phase 10 Streamlit Deployment Decision

## Recommendation

Keep Streamlit local for the immediate evaluation/interview deadline and point
it at the deployed Cloud Run API.

This is the most practical first release because:

- the backend demonstrates the production skills being evaluated: Docker,
  authentication, rate limiting, structured logging, health checks, and cloud
  deployment;
- the existing Streamlit application already communicates only through HTTP;
- local Streamlit avoids a second hosting account/service and another cold
  start;
- the backend key stays in the presenter's local environment;
- the UI can still demonstrate the real public backend during interviews.

## Minimum deployment configuration added

No UI layout or behavior changed. `src/app/streamlit_app.py` now:

- reads `API_BASE_URL` from the server-side environment;
- reads `API_KEY` first from the environment and then from `st.secrets`;
- sends `X-API-Key` only from Python server-side requests;
- does not send the key to public `/health` or `/ready`;
- supports `API_REQUEST_TIMEOUT_SECONDS`.

Local use:

```powershell
$env:API_BASE_URL = "https://YOUR_CLOUD_RUN_URL"
$env:API_KEY = "YOUR_BACKEND_API_KEY"
streamlit run src/app/streamlit_app.py
```

For a future hosted Streamlit service, store values in its protected secret
manager or `.streamlit/secrets.toml`:

```toml
API_KEY = "your-backend-key"
```

Never commit that file; `.gitignore` explicitly excludes it.

## When public Streamlit deployment is worth it

Deploy the UI only after:

1. the Cloud Run API is publicly verified;
2. the UI host supports server-side secrets;
3. the backend URL and key are configured outside source;
4. cold-start expectations are explained to demo users;
5. the added maintenance burden is acceptable.

Containerizing Streamlit is unnecessary for the first public backend
deployment. The current separation already demonstrates a clean client/API
architecture.
