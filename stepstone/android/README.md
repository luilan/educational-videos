# StepStone (Android)

A study companion for the lesson series in this repo. It downloads lesson packs (the same content, frames and
questions as the study-guide PDFs), walks through each lesson **one concept at a time**, and only unlocks the next
concept once you pass the current one (75% of its questions). Every answer is recorded: retake lessons, practise what
you missed, and a spaced review queue brings missed questions back after 1, 3 and 7 days.

**Download:** [StepStone.apk](https://github.com/luilan/educational-videos/releases/latest/download/StepStone.apk)
(Android 8.0+; allow installing from your browser or file manager when asked). The app tells you when a newer version
is out.

## How it works

- Lessons come from the catalog at `https://luilan.github.io/educational-videos/stepstone/catalog.json`, built from the
  study guides by `stepstone/build_packs.py` (format: `stepstone/FORMAT.md`). New series appear in the app as soon as
  they are published; updated packs show an *Updated* badge.
- Questions: multiple choice, true/false, numbers (with the precision the guide prints) and ordering are graded
  automatically; short answers and code are self-graded against the model answer.
- Videos stream from YouTube by default; a per-series switch downloads them for offline study.
- Progress stays on the phone; export/import it from the Progress tab.

## Build

```bash
./gradlew testDebugUnitTest assembleRelease
```

JDK 17 and the Android SDK (compileSdk 36). Release signing reads `/home/codex/secrets/stepstone/signing.properties`
(or the file named by `STEPSTONE_SIGNING`); without it the release build falls back to the debug key.
Tests: grading rules, the live catalog and packs (redirects, checksums, every lesson parses), and a Robolectric UI test
that downloads a pack and studies a full lesson, writing screenshots to `app/build/ui-shots/`.
