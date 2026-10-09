import hashlib
import json
import os
import re
import threading
import time

import requests
from flask import Flask, Response, jsonify, make_response, request

app = Flask(__name__)
BACKEND_URL = os.getenv(
    "SCORE_BACKEND_URL",
    "https://repository-name-cricket-score-backend-1.onrender.com/selected-score",
).strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TOKEN = os.getenv("COMMENTARY_TOKEN", "").strip()
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini").strip()
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin").strip()
MIN_GAP = max(0, int(os.getenv("COMMENTARY_MIN_GAP", "8")))
POLL_SECONDS = max(2, int(os.getenv("COMMENTARY_POLL_SECONDS", "5")))
lock = threading.Lock()
state = {}


def authorized():
    supplied = request.headers.get("X-Commentary-Token", "")
    cookie = request.cookies.get("commentary_auth", "")
    return bool(TOKEN and (supplied == TOKEN or cookie == TOKEN))


def match_id():
    value = str(request.args.get("match_id", "")).strip()
    return value if re.fullmatch(r"\d{3,12}", value) else ""


def safe(value, limit=120):
    return str(value or "").strip()[:limit]


def fetch_snapshot(mid):
    response = requests.get(
        BACKEND_URL,
        params={"match_id": mid, "_": str(time.time_ns())},
        headers={
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
            "User-Agent": "isolated-ai-cricket-commentary/1.0",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    match = payload.get("match") if isinstance(payload, dict) else None
    if not isinstance(match, dict):
        raise ValueError("Score backend returned no match snapshot.")

    batsmen = []
    for player in (match.get("batsmen") or [])[:2]:
        if isinstance(player, dict):
            batsmen.append({
                "name": safe(player.get("name"), 80),
                "runs": safe(player.get("runs"), 12),
                "balls": safe(player.get("balls"), 12),
                "striker": bool(player.get("striker")),
            })

    raw_bowler = match.get("bowler")
    bowler = None
    if isinstance(raw_bowler, dict):
        bowler = {
            key: safe(raw_bowler.get(key), 80 if key == "name" else 12)
            for key in ("name", "overs", "runs", "wickets", "economy")
        }

    raw_over = match.get("current_over") if isinstance(match.get("current_over"), dict) else {}
    balls = [
        {
            "label": safe(ball.get("label"), 12),
            "ball": safe(ball.get("ball"), 12),
            "over_ball": safe(ball.get("over_ball"), 12),
        }
        for ball in (raw_over.get("balls") or [])[-6:]
        if isinstance(ball, dict)
    ]
    batting_index = match.get("batting_index")
    if batting_index not in (0, 1):
        batting_index = None

    return {
        "match_id": mid,
        "title": safe(match.get("title"), 160),
        "team1": safe(match.get("team1"), 80),
        "team2": safe(match.get("team2"), 80),
        "team1_score": safe(match.get("team1_score"), 24),
        "team2_score": safe(match.get("team2_score"), 24),
        "team1_overs": safe(match.get("team1_overs"), 16),
        "team2_overs": safe(match.get("team2_overs"), 16),
        "batting_index": batting_index,
        "batting_team": safe(match.get("batting_team"), 80),
        "status": safe(match.get("status"), 160),
        "crr": safe(match.get("crr"), 16),
        "partnership": safe(match.get("partnership"), 24),
        "batsmen": batsmen,
        "bowler": bowler,
        "current_over": {
            "over": safe(raw_over.get("over"), 12),
            "balls": balls,
            "last_ball": safe(raw_over.get("last_ball"), 12),
            "free_hit": bool(raw_over.get("free_hit")),
        },
    }


def fingerprint(snapshot):
    raw = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def event_context(previous, current):
    if previous is None:
        return {
            "kind": "opening_snapshot",
            "note": "Give a short state-based introduction; do not claim a ball just happened.",
        }
    old_balls = previous.get("current_over", {}).get("balls", [])
    new_balls = current.get("current_over", {}).get("balls", [])
    score_changed = any(
        previous.get(key) != current.get(key)
        for key in ("team1_score", "team2_score", "team1_overs", "team2_overs")
    )
    players_changed = (
        previous.get("batsmen") != current.get("batsmen")
        or previous.get("bowler") != current.get("bowler")
    )
    if old_balls != new_balls:
        latest = new_balls[-1] if new_balls else {}
        return {
            "kind": "delivery_feed_changed",
            "verified_latest_ball_label": latest.get("label", ""),
            "verified_latest_ball_number": latest.get("ball") or latest.get("over_ball") or "",
            "note": "Use only the source label. Never invent shot, pace, field placement, or intent.",
        }
    return {
        "kind": "score_or_player_state_changed" if score_changed or players_changed else "state_changed",
        "note": "If the ball event is unclear, describe only verified state and do not guess what happened.",
    }


def openai_post(path, payload, timeout=45):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    response = requests.post(
        "https://api.openai.com" + path,
        headers={"Authorization": "Bearer " + OPENAI_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        app.logger.warning("OpenAI returned HTTP %s", response.status_code)
        raise RuntimeError("OpenAI request failed with HTTP " + str(response.status_code))
    return response


def generate_text(current, previous):
    instructions = (
        "You are a professional cricket analyst speaking naturally in Hinglish for a live broadcast. "
        "Return one concise spoken update, normally 1-2 sentences and under 45 words. Calm, sharp, analytical, "
        "broadcast-ready; no emojis, headings, markdown, or generic filler. Use only the supplied verified "
        "snapshot and change context. Never invent a delivery, player stat, score, target, required rate, result, "
        "or tactic. If data is stale/incomplete or there is no meaningful update, output exactly SKIP. Do not "
        "claim a ball just happened unless the delivery feed changed. Do not mention AI."
    )
    data = {
        "current_verified_snapshot": current,
        "previous_verified_snapshot": previous,
        "event_context": event_context(previous, current),
    }
    response = openai_post(
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "instructions": instructions,
            "input": json.dumps(data, ensure_ascii=False),
            "max_output_tokens": 130,
        },
    )
    chunks = []
    for item in response.json().get("output", []):
        if item.get("type") == "message":
            chunks.extend(
                part["text"]
                for part in item.get("content", [])
                if part.get("type") == "output_text" and part.get("text")
            )
    spoken = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    return "" if not spoken or spoken.upper() == "SKIP" else spoken[:700]


def synthesize(spoken_text):
    response = openai_post(
        "/v1/audio/speech",
        {
            "model": TTS_MODEL,
            "voice": TTS_VOICE,
            "input": spoken_text,
            "instructions": (
                "Natural clear Indian Hinglish, professional cricket broadcast, moderate pace, "
                "clean pronunciation. No music or sound effects."
            ),
            "response_format": "mp3",
        },
        timeout=60,
    )
    return response.content


@app.get("/health")
def health():
    return jsonify({
        "service": "isolated-ai-cricket-commentary",
        "status": "online",
        "score_backend_configured": bool(BACKEND_URL),
        "openai_configured": bool(OPENAI_API_KEY),
        "token_configured": bool(TOKEN),
    })


@app.get("/")
def player():
    page = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;width:100%;height:100%;background:transparent;overflow:hidden}
#enable{display:none;position:fixed;left:12px;bottom:12px;border:0;border-radius:8px;padding:10px 14px;background:#101820;color:white;font:600 14px Arial}
</style></head><body>
<button id="enable">Enable commentary audio</button><audio id="audio" preload="none"></audio>
<script>
const pollMs=__POLL_MS__, mid=new URLSearchParams(location.search).get("match_id")||"";
const audio=document.getElementById("audio"), btn=document.getElementById("enable");
let enabled=true,busy=false;
async function poll(){
 if(!enabled||busy||!mid)return; busy=true;
 try{
  const r=await fetch("/next.mp3?match_id="+encodeURIComponent(mid)+"&_="+Date.now(),{cache:"no-store"});
  if(r.status===200){
   const url=URL.createObjectURL(await r.blob()); audio.src=url;
   try{await audio.play();}catch(e){URL.revokeObjectURL(url);btn.style.display="block";enabled=false;busy=false;return;}
   audio.onended=()=>{URL.revokeObjectURL(url);busy=false;setTimeout(poll,250);};
   audio.onerror=()=>{URL.revokeObjectURL(url);busy=false;setTimeout(poll,pollMs);};
   return;
  }
 }catch(e){}
 busy=false;setTimeout(poll,pollMs);
}
btn.onclick=()=>{enabled=true;btn.style.display="none";poll();};
if(mid)poll();else{btn.textContent="Add match_id to commentary URL";btn.style.display="block";}
</script></body></html>"""
    response = make_response(page.replace("__POLL_MS__", str(POLL_SECONDS * 1000)))
    # The player uses a same-origin HttpOnly cookie, so the token is not exposed in page JavaScript.
    response.set_cookie(
        "commentary_auth",
        TOKEN,
        httponly=True,
        secure=request.is_secure,
        samesite="Lax",
        max_age=60 * 60 * 12,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/status")
def status():
    if not authorized():
        return jsonify({"error": "unauthorized or COMMENTARY_TOKEN is not configured"}), 401
    mid = match_id()
    if not mid:
        return jsonify({"error": "numeric match_id is required"}), 400
    try:
        return jsonify({"ok": True, "match_id": mid, "snapshot": fetch_snapshot(mid)})
    except Exception as exc:
        app.logger.warning("Snapshot fetch failed: %s", str(exc)[:180])
        return jsonify({"ok": False, "error": str(exc)[:180]}), 502


@app.get("/next.mp3")
def next_mp3():
    if not authorized():
        return jsonify({"error": "unauthorized or COMMENTARY_TOKEN is not configured"}), 401
    mid = match_id()
    if not mid:
        return jsonify({"error": "numeric match_id is required"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is not configured"}), 503

    with lock:
        try:
            current = fetch_snapshot(mid)
        except Exception as exc:
            app.logger.warning("Score snapshot fetch failed: %s", str(exc)[:180])
            return Response(status=204, headers={"Cache-Control": "no-store"})

        fp = fingerprint(current)
        old_state = state.get(mid)
        if old_state and old_state.get("fingerprint") == fp:
            return Response(status=204, headers={"Cache-Control": "no-store"})
        now = time.time()
        if old_state and now - old_state.get("last_spoken_at", 0) < MIN_GAP:
            return Response(status=204, headers={"Cache-Control": "no-store"})

        previous = old_state.get("snapshot") if old_state else None
        status_text = current.get("status", "").lower()
        if (
            not current.get("team1")
            or not current.get("team2")
            or "retry" in status_text
            or "temporarily unavailable" in status_text
        ):
            state[mid] = {
                "fingerprint": fp,
                "snapshot": current,
                "last_spoken_at": old_state.get("last_spoken_at", 0) if old_state else 0,
            }
            return Response(status=204, headers={"Cache-Control": "no-store"})

        try:
            spoken_text = generate_text(current, previous)
            if not spoken_text:
                state[mid] = {
                    "fingerprint": fp,
                    "snapshot": current,
                    "last_spoken_at": old_state.get("last_spoken_at", 0) if old_state else 0,
                }
                return Response(status=204, headers={"Cache-Control": "no-store"})
            audio_bytes = synthesize(spoken_text)
        except Exception as exc:
            app.logger.warning("Commentary generation failed: %s", str(exc)[:180])
            return Response(status=204, headers={"Cache-Control": "no-store"})

        state[mid] = {"fingerprint": fp, "snapshot": current, "last_spoken_at": time.time()}
        return Response(
            audio_bytes,
            mimetype="audio/mpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Commentary-Mode": "voice-only",
            },
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
