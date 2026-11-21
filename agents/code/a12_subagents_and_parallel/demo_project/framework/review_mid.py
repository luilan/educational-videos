"""Contact sheet of frames at 35% and 70% through every section: python review_mid.py vNN out.png"""
import glob
import json
import sys

import av
from PIL import Image

video, out = sys.argv[1], sys.argv[2]
path = sorted(glob.glob(f"media/videos/{video}_scene/1080p60/*.mp4"))[0]
secs = json.load(open(f"voice/{video}/sections.json"))
t, want = 7.7, []
for s in secs:
    want += [t + 0.35 * s["dur"], t + 0.7 * s["dur"]]
    t += s["dur"] + 0.5
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
print(out, len(ims), "frames")
