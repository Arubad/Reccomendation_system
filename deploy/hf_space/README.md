---
title: HookSeq demo
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: 3.10
pinned: false
---

# HookSeq demo

Next-track recommendations from a listening session on Yambda: HookSeq (audio, artist, event
and organic signals) next to a plain SASRec, plus a "new track audience" tab for tracks with
audio but no ID embedding.

Pinned: torch 2.8.0 (ZeroGPU-supported) and gradio 6.31.0; `hookrec` is installed by `app.py`
without its training dependencies, at the tag set in `HOOKREC_REF`.

Hardware: this Space is meant for **ZeroGPU** (choose it in Settings, Space hardware). Models
load on CPU at startup; scoring runs inside `spaces.GPU` calls. It also runs on CPU hardware.

Set the `HOOKSEQ_BUNDLE_REPO` variable (Settings, Variables) to the Hugging Face model repo
holding the demo bundle, or place the bundle in `bundle/` next to `app.py`.
Setup details: `docs/hf_space.md` in the code repository.
