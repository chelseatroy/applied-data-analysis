# Getflix Session 8 Dashboard

Instructor-facing live dashboard for the Getflix ML operationalization activity. Polls the class Google Sheet for student ngrok URLs and queries every student's inference server in real time.

## Deployed app

**URL:** https://getflix-inference-dashboard.onrender.com

**Render admin** (instructor only — manage deploys, logs, env vars): https://dashboard.render.com

## What it does

Four tabs:

| Tab | What it shows |
|-----|---------------|
| **Live** | Auto-refreshing table of every student's server — cluster model, rec model, cluster assignment for a given user, top 5 recommendations, genre forecast |
| **Compare** | Same view but on demand — enter any user ID to compare all servers at once |
| **Ensemble** | Select a subset of servers and get a consensus recommendation list, ranked by how many servers agreed |
| **Rate & Rec** | Rate movies yourself (1–5 stars); each server finds your nearest neighbour and recommends from there |

Student servers must expose: `/health`, `/cluster`, `/recommend`, `/forecast`, `/movies`, `/recommend_new_user`.

## Running locally

```bash
pip install flask requests
python session_8_dashboard.py
# Open http://localhost:5055
```

Optional flags:

```
--port      Port to serve on (default: 5055)
--interval  Browser auto-refresh interval in seconds (default: 15)
--user-id   Default user ID shown on Live and Compare tabs (default: 42)
--genre     Default genre for the forecast column (default: Drama)
--timeout   Per-student-server request timeout in seconds (default: 4)
```

## Deploying to Render

1. Push this repo to GitHub (the `dashboard/` folder is not gitignored).
2. Create a new **Web Service** on [render.com](https://render.com).
3. Connect the repo and set:
   - **Root directory:** `week_7/getflix_challenge/dashboard`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python session_8_dashboard.py`
4. Render injects `$PORT` automatically — the app reads it, so no port config needed.
5. Update the deployed URL at the top of this README.

## Student setup

Before class, share the Google Sheet link and ask students to paste their base URL into the `URL` column next to their name. The dashboard picks up new entries on the next refresh.
