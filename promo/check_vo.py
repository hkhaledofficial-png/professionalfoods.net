#!/usr/bin/env python3
"""QA a voice-over clip: ask Gemini to transcribe it and judge the accent.
Usage: python3 check_vo.py <clip.wav> [...]"""
import base64, json, os, sys, urllib.request

for path in sys.argv[1:]:
    data = base64.b64encode(open(path, "rb").read()).decode()
    body = {"contents": [{"parts": [
        {"inlineData": {"mimeType": "audio/wav", "data": data}},
        {"text": "Transcribe this Arabic audio exactly (Arabic script). Then on a new line, say in English: "
                 "is the accent Egyptian (Cairene) or other, is there any extra speech beyond the main sentence, "
                 "any mispronunciations, and timestamps where speech starts/ends. Be brief."}]}]}
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/" + os.environ.get("CHECK_MODEL", "gemini-3.5-flash") + ":generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    print("==", os.path.basename(path))
    print(d["candidates"][0]["content"]["parts"][0]["text"].strip())
