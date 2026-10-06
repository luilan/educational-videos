# StepStone content format, version 1

StepStone is a study companion app: it downloads lesson packs, walks through each lesson concept by concept, and
only unlocks the next concept once the learner has shown they understood the current one. The format is
subject-agnostic: nothing in it is specific to machine learning.

## Catalog

`https://luilan.github.io/educational-videos/stepstone/catalog.json`

```json
{"format": 1, "name": "…", "packs": [
  {"id": "deep-dive", "title": "How LLMs Work: Deep Dive", "description": "…", "subject": "Machine learning",
   "order": 4, "requires": ["how-llms-work"], "version": 2, "content_hash": "51b8bed7a6734a70",
   "file": "deep-dive-v2.zip", "url": "https://github.com/…/releases/download/stepstone-packs/deep-dive-v2.zip",
   "size": 18012345, "sha256": "…", "lessons": 45}]}
```

The app checks the catalog periodically. A pack is new if its `id` is unknown, updated if `version` is higher than
the installed one. Verify `sha256` after download. `requires` is advisory (shown as "start with …").

## Pack (ZIP)

```
manifest.json
lessons/<lesson id>.json
img/<file>.jpg
cover.jpg            (optional)
```

`manifest.json`: `format`, `id`, `title`, `description`, `subject`, `language`, `order`, `requires`, `source`,
`cover` (path or null), `content_hash`, and `lessons`: a list of
`{id, number, title, tagline, duration, file, concepts, exercises, video}`.

## Lesson

```json
{"format": 1, "id": "d45", "number": 45, "label": "Episode 45", "title": "Circuits", "tagline": "…",
 "duration": "2:16", "intro": [blocks], "prereq": [blocks],
 "video": {"youtube": "<id or null>", "offline": {"url": "…", "size": 1234, "sha256": "…"} or null},
 "concepts": [{"id": "c1", "title": "…", "segment": [8, 34], "blocks": [blocks], "exercises": [exercises]}],
 "links": {"study_pdf": "…", "code": "…"}}
```

`segment` is the concept's start and end in the video, in seconds: "watch this part" seeks there (YouTube `t=` or
the offline file).

## Blocks

| type | fields | notes |
|---|---|---|
| `text` | `html`, optional `align: "center"` | inline HTML limited to `b i code sub sup br` |
| `callout` | `style` (`key` or `note`), `title`, `html` | `key` = the concept's key idea |
| `figures` | `items: [{src, caption, t}]` | `src` is a path inside the pack; `t` = video time of the frame |
| `table` | `header` (list or null), `rows` (list of lists) | cells are inline HTML |
| `code` | `language`, `text` | monospace, horizontally scrollable |
| `html` | `html` | fallback for layouts the other blocks can't express; render in a WebView |

Unknown block types must be skipped with a notice ("update the app to see this part"), never crash.

## Exercises

```json
{"id": "c2e1", "kind": "number", "prompt": [blocks], "grading": "auto",
 "key": {"parts": [{"label": null, "value": 0.28, "tol": 0.005, "unit": null}]},
 "answer": [blocks], "why": [blocks]}
```

| kind | extra fields | key (when `grading` is `auto`) | auto-grading |
|---|---|---|---|
| `mc` | `options` (inline HTML) | `{"choice": i}` | right if the chosen index is `i` |
| `tf` | | `{"value": true/false}` | |
| `number` | | `{"parts": [{label, value, tol, unit}]}` | one input per part; right if `|x − value| ≤ tol`; with unit `%` also accept the fraction (`x × 100`) |
| `order` | | `{"items": [...]}` in the correct order | show the items shuffled; right if the learner's order matches |
| `short` | | none | self-graded |
| `code` | `code` | none | self-graded; show the code with a copy button |

`grading: "self"`: show the model answer, the learner marks *got it* (1), *partly* (0.5) or *missed* (0).
Unknown kinds: treat as self-graded.

Number input should accept `,` as a thousands separator, a leading `−`/`-`, and a trailing `%`.

## Mastery gate (app behaviour, not data)

A concept is passed when its exercise score (auto: 1 or 0; self: 1, 0.5, 0) reaches 75% of its exercises. If not,
the app shows the concept's key idea again and re-asks only the missed exercises. The next concept unlocks after a
pass; passed concepts stay open for review.

## Authoring

Packs are built from the study guides by `stepstone/build_packs.py`; answer keys live in the study files
(`"key"` on each exercise) and are validated by `framework/answer_keys.py check`. `stepstone/publish.py` uploads
offline videos and packs to GitHub releases and writes the catalog.
