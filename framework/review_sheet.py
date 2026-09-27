"""Contact sheet of section-end frames: python review_sheet.py vNN out.png [extra_seconds...]"""
import json
import sys

import av
from PIL import Image

video, out = sys.argv[1], sys.argv[2]
extra = [float(x) for x in sys.argv[3:]]
path = sorted(__import__("glob").glob(f"media/videos/{video}_scene/1080p60/*.mp4"))[0]
secs = json.load(open(f"voice/{video}/sections.json"))
t, want = 7.7, [4.5]
for s in secs:
    t += s["dur"] + 0.5
    want.append(t - 0.6)
want += extra
c = av.open(path)
idx = {int(x * 60): k for k, x in enumerate(want)}
ims = {}
for i, fr in enumerate(c.decode(video=0)):
    if i in idx:
        ims[idx[i]] = fr.to_image().resize((480, 270))
        if len(ims) == len(idx):
            break
cols = 4
rows = (len(want) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 490, rows * 280), "white")
for k, im in ims.items():
    sheet.paste(im, ((k % cols) * 490, (k // cols) * 280))
sheet.save(out)
print(path, f"{float(c.duration / av.time_base):.1f}s", len(ims), "frames")
