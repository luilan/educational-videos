# Educational Videos

Source code for animated, narrated explainer videos in the style of 3Blue1Brown, made with
[Manim Community](https://www.manim.community/) v0.21.

| Series | Videos | Length | About |
|---|---|---|---|
| [Foundations](foundations/) | 14 (F01–F14) | ~27 min | The math and code behind LLMs: vectors, dot products, matrices, nonlinearity, exp/log, probability, softmax, statistics, waves, gradients, NumPy, neural nets, autograd, train/validation data |
| [How LLMs Work](how-llms-work/) | 14 (1–14) | ~32 min | A large language model from scratch: tokens → embeddings → attention → transformer blocks → training → a real tiny GPT → chatbots |
| [LLMs in Practice](llms-in-practice/) | 12 (P01–P12) | ~30 min | Building with LLMs: prompts, context, sampling, embeddings, RAG, tool use, agents, LoRA, quantization, evaluation, safety. Every episode ships a study guide and runnable code |
| [Deep Dive](deep-dive/) | ~45 planned | in production | Inside a modern LLM, piece by piece: BPE, RoPE, attention variants, normalization, training and scaling, MoE, inference tricks, RLHF/DPO, interpretability. Every episode ships a study guide and runnable code |

Foundations is an optional pre-series: each video ends by pointing to the main episodes that use it.

## Repository layout

```
framework/          shared code used by every video
  voiced_scene.py     VoicedScene: plays narration sections and cues animations to spoken words
  intro.py            the "token" title intro used at the start of every video
  common.py           shared visuals: token boxes, code panels, end cards
  voice.py            narration generator (Kokoro TTS + Whisper word timestamps)
  review_sheet.py     contact sheet of section-end frames, for QA
  review_mid.py       contact sheet of mid-section frames, for QA
  intro_styles.py     the five intro styles that were tried
  study_guide.py      builds a lesson's study-guide PDF from <series>/study/<id>_study.py
  download_models.sh  fetches the TTS model into tts/
foundations/        pre-series: fNN_script.py + fNN_scene.py per video (+ study/)
how-llms-work/      main series: vNN_script.py + vNN_scene.py per episode
  study/              one study-guide PDF per episode (+ its source and video frames)
  tiny_gpt/           the ~100-line GPT trained for episodes 11–12
render.sh           render one video
narrate.sh          (re)generate one video's narration
```

## How a video is made

1. **Script**: `<series>/<id>_script.py` holds the title, tagline and narration, one string per section.
2. **Narration**: `./narrate.sh <series> <id>` synthesizes each section with the Kokoro `af_heart` voice and
   transcribes it with faster-whisper to get a timestamp for every word. Output: `voice/<id>/s<N>.wav` plus `sections.json`.
3. **Scene**: `<series>/<id>_scene.py` subclasses `VoicedScene`. It calls `self.section(i)` to start section *i*'s audio,
   `self.at("word")` to wait until that word is spoken, and `self.end_section()` to wait for the section to finish.
   Rendering prints a warning if animations run longer than their narration.
4. **Render**: `./render.sh <series> <id>` (1080p60) or `./render.sh <series> <id> -ql` (fast 480p draft).

## Study guides

Every video has a study-guide PDF in `<series>/study/<id>_study.pdf` (28 guides, about 300 pages, 670 questions).
Each guide splits the lesson into concepts, to be mastered one at a time. Each concept has a short explanation,
a frame from the video with its timestamp, a key idea and a check of 3–4 questions (multiple choice, true/false,
calculations, ordering and hands-on code). Answers with explanations are at the end. Every numeric answer and code
output was checked by running it.

The content lives in `<series>/study/<id>_study.py`. To rebuild a PDF (needs Google Chrome or Chromium and `pypdf`;
frames are cached in `study/img/`, so the rendered video is only needed for new figures):

```bash
python framework/study_guide.py how-llms-work v05
```

## Setup

```bash
# System libraries for Manim (Ubuntu/Debian)
sudo apt-get install -y pkg-config libcairo2-dev libpango1.0-dev python3-dev

python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU PyTorch (tiny GPT, F13)
pip install -r requirements.txt

framework/download_models.sh        # Kokoro TTS model, ~340 MB, into tts/ (git-ignored)
```

LaTeX is **not** required: all text and math on screen uses Manim's `Text` (no `MathTex`/`Tex`).
The first `./narrate.sh` run also downloads the Whisper `small.en` model.

```bash
./narrate.sh how-llms-work v01     # only needed once per video (audio is git-ignored)
./render.sh  how-llms-work v01     # → how-llms-work/media/videos/v01_scene/1080p60/WhatAnLLMDoes.mp4
```

## Notes

- Rendered videos (`media/`) and narration audio (`*.wav`) are not committed. `sections.json` is kept because the
  scenes' cue words depend on it. Kokoro output is deterministic, so regenerated audio matches it.
- Made-up example numbers are captioned "illustrative" on screen. Real figures (GPT-2 token IDs and sizes, the tiny
  GPT's parameters and losses) were checked against real tokenizers, code and training runs.
- Text-to-speech pronunciation fixes live in the scripts themselves (e.g. "byte pair, encoding", "Pre-fill").
