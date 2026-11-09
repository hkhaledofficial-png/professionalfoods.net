# Professional Ready Mix — Brand promo videos

Three short vertical (9:16, 1080×1920) promos for **Professional Ready Mix**
(بروفيشنال ريدي ميكس). Their job is brand identity and presence. They use the same
art direction as the Professional Foods promos (dark premium base, one strong accent
color, spaced-caps English kickers, cinematic grade, short Arabic headlines), but they
use the Ready Mix logo and its blue (`#3075FD`).

| # | File | Theme |
|---|------|-------|
| 1 | `output/professional-ready-mix-01-strength.mp4` | Strength and fleet: "كل مشروع كبير بيبدأ من أساس متين" |
| 2 | `output/professional-ready-mix-02-projects.mp4` | Projects in West Cairo (6th of October, Sheikh Zayed, Giza) plus major government projects |
| 3 | `output/professional-ready-mix-03-reach.mp4` | Beyond Cairo and Giza: work in many governorates |

Each video has:
- An Egyptian Arabic voice-over generated with Gemini TTS, voice `Algenib`
- Arabic on-screen headlines (Cairo font) and English kickers (Inter)
- An original synthesized music bed, ducked under the voice
- A small white logo mark at the top of footage scenes and an animated logo reveal at the end

## Re-rendering

```bash
export GEMINI_API_KEY=...        # never commit this
python3 render.py --media <dir containing v/v1..v6.mp4 and imgs/> \
                  --fonts <dir with Cairo-Black/Bold, Inter-ExtraBold/SemiBold, Tajawal-Medium .ttf> \
                  --out output --preview
```

- The scripts (scenes, on-screen text, voice-over lines) live in `build()` in `render.py`.
- Each scene stretches to fit its voice-over line.
- Voice-over clips are cached in `<out>/.vo/`.
- `check_vo.py` transcribes a clip with Gemini to check the wording and the accent.
- Several of the source clips were filmed with the phone on its side. `render.py`
  rotates them back (`transpose=2`).
