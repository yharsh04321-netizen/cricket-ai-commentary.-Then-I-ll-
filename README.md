# AI Cricket Commentary (separate service)

This repository contains an isolated voice-only Hinglish cricket commentary service for OBS. It reads the existing score backend and does not modify that backend or its scorecard.

## Render deployment

Create a **new** Render Web Service connected to this repository:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4`

Set these environment variables in the new Render service:
- `OPENAI_API_KEY` — keep secret; never commit it.
- `COMMENTARY_TOKEN` — long random secret.
- `SCORE_BACKEND_URL` — `https://repository-name-cricket-score-backend-1.onrender.com/selected-score`

Optional: `OPENAI_TEXT_MODEL` (default `gpt-4.1-mini`), `OPENAI_TTS_MODEL` (default `gpt-4o-mini-tts`), `OPENAI_TTS_VOICE` (default `marin`), `COMMENTARY_MIN_GAP` (default `8`), `COMMENTARY_POLL_SECONDS` (default `5`).

## OBS

Add the new service's player URL as a **separate Browser Source**:
`https://YOUR-COMMENTARY-SERVICE.onrender.com/?match_id=YOUR_MATCH_ID`

Keep your existing scorecard source unchanged. Use the same match ID. If audio autoplay is blocked, click **Enable commentary audio** once in the browser source properties.

## Important

- The service only reads `/selected-score?match_id=...`; it does not send writes to the score backend.
- Test the `/health` and `/status?match_id=...` endpoints before using it live.
- OpenAI API usage is billed separately.
