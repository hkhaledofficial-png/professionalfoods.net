#!/usr/bin/env python3
"""Egyptian-Arabic voice-over lines via Gemini TTS.

Reads the API key from the GEMINI_API_KEY env var (never stored in the repo).
Usage: python3 tts.py <out.wav> "<line>" [voice] [model]
"""
import base64, json, os, sys, urllib.error, urllib.request, wave

STYLE = ("Speak ONLY the transcript below, nothing else. Voice: confident, warm, premium male TV-commercial "
         "voice-over with a natural Cairene Egyptian Arabic accent, measured cinematic pace, clear articulation.\n\n"
         "TRANSCRIPT:\n")


def tts(text, out, voice="Algenib", model="gemini-3.8-flash-tts"):
    keys = [k for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k]
    body = {
        "contents": [{"parts": [{"text": STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    mk = lambda key: urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    ki = 0
    import time
    for attempt in range(12):
        try:
            with urllib.request.urlopen(mk(keys[ki % len(keys)]), timeout=180) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 11:
                raise
            if e.code == 429 and len(keys) > 1:
                ki += 1  # rotate to the next key; only sleep after a full cycle
                if ki % len(keys):
                    continue
            time.sleep(25)  # free tier: ~3 requests/minute
    part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
    pcm = base64.b64decode(part["data"])
    rate = 24000
    mt = part.get("mimeType", "")
    if "rate=" in mt:
        rate = int(mt.split("rate=")[1].split(";")[0])
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out


if __name__ == "__main__":
    tts(sys.argv[2], sys.argv[1], *(sys.argv[3:5]))
