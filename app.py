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
<title>AI Cricket Commentary</title>
<style>
*{box-sizing:border-box}html,body{margin:0;min-height:100%;font-family:Arial,sans-serif;background:#0b1220;color:#f8fafc}
body{padding:18px}.panel{max-width:620px;margin:0 auto;border:1px solid #293449;border-radius:16px;background:#111b2d;padding:20px;box-shadow:0 12px 36px #0004}
h1{font-size:21px;margin:0 0 7px}.sub{color:#aab8ce;font-size:14px;line-height:1.5}
.row{display:flex;gap:10px;margin:16px 0 10px}input{min-width:0;flex:1;background:#080f1c;color:#fff;border:1px solid #42516b;border-radius:9px;padding:12px;font-size:16px}
button{border:0;border-radius:9px;padding:12px 15px;background:#22c55e;color:#07130b;font-size:15px;font-weight:700;cursor:pointer}
#start{width:100%;margin:8px 0 12px}#status{padding:12px;border-radius:9px;background:#0a1322;color:#cbd5e1;font-size:14px;line-height:1.5}
#score{font-size:15px;font-weight:700;margin-top:14px;line-height:1.6}.muted{font-size:12px;color:#93a4bc;margin-top:12px}
</style></head><body>
<main class="panel">
<h1>🎙️ AI Cricket Commentary</h1>
<div class="sub">Live score feed check and voice commentary. Choose the match ID, then tap Start Commentary so your phone/browser permits audio.</div>
<div class="row"><input id="mid" inputmode="numeric" autocomplete="off" placeholder="Match ID (numbers only)"><button id="load">Load</button></div>
<button id="start">▶ Start Commentary</button>
<div id="status" role="status">Ready. Enter a match ID and tap Load.</div>
<div id="score"></div>
<div class="muted">This service only reads the existing score backend. It does not change your scorecard or match selector.</div>
<audio id="audio" preload="none"></audio>
</main>
<script>
const pollMs=__POLL_MS__, params=new URLSearchParams(location.search);
const midInput=document.getElementById("mid"), statusEl=document.getElementById("status"), scoreEl=document.getElementById("score");
const startBtn=document.getElementById("start"), loadBtn=document.getElementById("load"), audio=document.getElementById("audio");
let mid=params.get("match_id")||"", running=false, busy=false, timer=null;
midInput.value=mid;
function say(s){statusEl.textContent=s}
function selectedId(){return /^\d{3,12}$/.test(midInput.value.trim())?midInput.value.trim():""}
async function loadStatus(){
 if(!mid){say("Enter a numeric match ID, then tap Load.");return false}
 let lastMessage="The score backend is taking too long to respond.";
 for(let attempt=1;attempt<=3;attempt++){
  try{
   say(attempt===1?"Checking score feed…":"Score backend is slow; retrying automatically ("+attempt+"/3)…");
   const r=await fetch("/status?match_id="+encodeURIComponent(mid)+"&_="+Date.now(),{cache:"no-store"});
   const data=await r.json().catch(()=>({}));
   if(r.ok){
    const s=data.snapshot||{};
    scoreEl.textContent=(s.title?s.title+" — ":"")+
     (s.team1||"Team 1")+" "+(s.team1_score||"")+
     "  |  "+(s.team2||"Team 2")+" "+(s.team2_score||"")+
     (s.status?"\n"+s.status:"");
    say("Score backend connected. Match snapshot loaded.");
    return true;
   }
   lastMessage="HTTP "+r.status+": "+(data.error||"Score backend did not return a snapshot.");
   if(r.status!==502 && r.status!==503){break}
  }catch(e){lastMessage="The commentary service could not complete the score request."}
  if(attempt<3)await new Promise(resolve=>setTimeout(resolve,2000));
 }
 scoreEl.textContent="";
 say("Score feed unavailable after retries. "+lastMessage+" Try Load again in 20–30 seconds; the existing score backend may be waking up or responding slowly.");
 return false;
}
async function poll(){
 if(!running||busy||!mid)return;
 busy=true;
 try{
  const r=await fetch("/next.mp3?match_id="+encodeURIComponent(mid)+"&_="+Date.now(),{cache:"no-store"});
  if(r.status===200){
   const blob=await r.blob(), url=URL.createObjectURL(blob);
   audio.src=url;
   try{await audio.play();say("🎙️ Playing AI commentary.");}
   catch(e){URL.revokeObjectURL(url);say("Audio is blocked by the browser. Tap Start Commentary again.");running=false;startBtn.textContent="▶ Start Commentary";busy=false;return}
   audio.onended=()=>{URL.revokeObjectURL(url);busy=false;setTimeout(poll,250)};
   audio.onerror=()=>{URL.revokeObjectURL(url);busy=false;say("Audio playback failed. Tap Start Commentary to retry.");setTimeout(poll,pollMs)};
   return;
  }
  if(r.status===401){say("Authentication is not set. Open the main service URL again; if this persists, check COMMENTARY_TOKEN in Render.");running=false}
  else if(r.status===503){say("OpenAI API key is not configured. Check OPENAI_API_KEY in the commentary service's Render environment.");running=false}
  else if(r.status!==204){say("Commentary request returned HTTP "+r.status+". Check the commentary service logs.");}
  else {say("Connected — waiting for a new score or player update to commentate.");}
 }catch(e){say("Connection hiccup. Retrying automatically…")}
 busy=false;
 if(running)setTimeout(poll,pollMs);
}
loadBtn.onclick=async()=>{mid=selectedId();if(!mid){say("Please enter a numeric match ID.");return}history.replaceState(null,"","?match_id="+encodeURIComponent(mid));await loadStatus()};
startBtn.onclick=async()=>{
 if(!mid){mid=selectedId();if(!mid){say("Enter a numeric match ID first.");return}history.replaceState(null,"","?match_id="+encodeURIComponent(mid))}
 const ok=await loadStatus();if(!ok)return;
 running=true;startBtn.textContent="■ Stop Commentary";say("Starting voice commentary…");poll();
 startBtn.onclick=()=>{running=false;startBtn.textContent="▶ Start Commentary";say("Commentary stopped. Tap Start Commentary to resume.")};
};
if(mid){loadStatus()}else{say("Enter a match ID above. The page now shows status instead of staying blank.")}
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
