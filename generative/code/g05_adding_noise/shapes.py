"""Shapes: the small synthetic image dataset used throughout How Generative Models Work.

Each image is 32×32 RGB with one colored shape (circle, square or triangle) on a dark background, at one of nine
positions, with a little random jitter in position and size. Each image comes with a caption such as
"a red circle at the top left". No downloads, no licenses: the whole dataset is made by this file.

    from shapes import make_dataset
    images, captions, labels = make_dataset(20000, seed=0)   # uint8 (N, 32, 32, 3), list of str, (N, 3) ints
"""
import numpy as np

SIZE = 32
BACKGROUND = (20, 22, 30)
SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (230, 60, 50), "green": (60, 200, 80), "blue": (60, 110, 240),
          "yellow": (240, 210, 50), "purple": (170, 80, 220)}
COLOR_NAMES = list(COLORS)
POSITIONS = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right"]


def draw(shape, color, position, rng):
    """One image (uint8 32×32×3) with a hard-edged shape; position is an index into POSITIONS."""
    img = np.empty((SIZE, SIZE, 3), np.uint8)
    img[:] = BACKGROUND
    row, col = divmod(position, 3)
    cy = 7 + 9 * row + rng.integers(-1, 2)           # centers at 7, 16, 25 (± 1 pixel)
    cx = 7 + 9 * col + rng.integers(-1, 2)
    r = rng.uniform(4.0, 5.5)                         # "radius" in pixels
    y, x = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    if shape == "circle":
        mask = (x - cx - 0.5) ** 2 + (y - cy - 0.5) ** 2 <= r * r
    elif shape == "square":
        h = r * 0.85
        mask = (abs(x - cx - 0.5) <= h) & (abs(y - cy - 0.5) <= h)
    else:  # triangle pointing up
        r *= 1.2
        top, bottom = cy + 0.5 - r, cy + 0.5 + r * 0.8
        half = (y - top) / (bottom - top) * r
        mask = (y >= top) & (y <= bottom) & (abs(x - cx - 0.5) <= half)
    img[mask] = COLORS[color]
    return img


def caption(shape, color, position):
    return f"a {color} {shape} at the {POSITIONS[position]}" if POSITIONS[position] != "center" \
        else f"a {color} {shape} in the center"


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    images = np.empty((n, SIZE, SIZE, 3), np.uint8)
    labels = np.stack([rng.integers(0, len(SHAPES), n), rng.integers(0, len(COLORS), n),
                       rng.integers(0, len(POSITIONS), n)], axis=1)
    captions = []
    for i, (s, c, p) in enumerate(labels):
        images[i] = draw(SHAPES[s], COLOR_NAMES[c], p, rng)
        captions.append(caption(SHAPES[s], COLOR_NAMES[c], p))
    return images, captions, labels


def save_grid(images, path, cols=8, scale=4, pad=2):
    """Save images (N, 32, 32, 3) as one PNG grid, each image scaled up with sharp pixels."""
    from PIL import Image
    n = len(images)
    rows = (n + cols - 1) // cols
    s = SIZE * scale
    out = Image.new("RGB", (cols * (s + pad) - pad, rows * (s + pad) - pad), (0, 0, 0))
    for i, im in enumerate(images):
        tile = Image.fromarray(np.asarray(im, np.uint8)).resize((s, s), Image.NEAREST)
        out.paste(tile, ((i % cols) * (s + pad), (i // cols) * (s + pad)))
    out.save(path)


if __name__ == "__main__":
    images, captions, _ = make_dataset(16, seed=1)
    save_grid(images, "shapes_sample.png")
    print("\n".join(captions))
    print("distinct captions:", len(SHAPES) * len(COLORS) * len(POSITIONS))
