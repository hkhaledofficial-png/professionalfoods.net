# Diamond SC album kit

Builds a 4:5 Instagram/Facebook photo album from a folder of match photos, in the same look
as the training / warm-up albums in `albums/`.

## Setup (once)
    pip install pillow numpy onnxruntime

The AI background-removal model (~176 MB) downloads itself on first run into `album-kit/models/`.

## Workflow
1. Contact sheet to pick photos:
       python album-kit/make_album.py "D:\elgezeera\الاول" out\first-half --contact
2. Build the album (cover + all photos, bright grade by default):
       python album-kit/make_album.py "D:\elgezeera\الاول" out\first-half --cover IMG_1234.JPG --word "FIRST HALF" --sub "DIAMOND|SC" --tag "MATCH DAY" --y 150
   - `--files a.jpg,b.jpg,...`  choose / order the album photos
   - `--y`          vertical position of the big word (lower it until the head overlaps the letters)
   - `--cover-top`  0..1 which part of a tall photo the 4:5 cover crop keeps (0 = top)
   - `--only-cover` re-render just the cover while tuning
   - `|` in `--word` / `--sub` makes a line break

## House style
- Colours: matched to the photographer's training edit; bright, never darkened; skin stays natural.
- Cover: huge condensed word (Anton) in Diamond yellow-gold `#F7C414` BEHIND the player,
  cream secondary text bottom-left (League Gothic), small tag next to it, crest bottom-right.
- Output: 1080x1350 JPG, `01-cover.jpg`, `02.jpg`, ...
