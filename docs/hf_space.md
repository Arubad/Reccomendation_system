# Deploy the demo to Hugging Face Spaces (ZeroGPU)

The demo reads a self-contained bundle (under 200 MB, no processed data needed), so the Space
only needs the bundle and the `hookrec` package.

1. **Export the bundle** from real checkpoints (the `model_final.pt` of the HookSeq and SASRec
   test runs, e.g. from the K4 outputs) and the matching processed dir:

       python scripts/export_demo.py --processed PROCESSED_DIR \
           --hookseq-ckpt ckpt/hookseq/test --sasrec-ckpt ckpt/a0_sasrec/test --out bundle/

   Try it locally first: `python -m hookrec.demo.app --bundle bundle/` (needs
   `pip install -e ".[demo]"`).
2. **Upload the bundle** to a Hugging Face *model* repo, for example
   `hf upload YOU/hookseq-demo-bundle bundle/ . --repo-type model`. A private repo needs an
   `HF_TOKEN` secret on the Space.
3. **Create a Space**: SDK Gradio, hardware **ZeroGPU**. New free accounts only get ZeroGPU; the
   demo also runs on CPU hardware.
4. **Set the variable** `HOOKSEQ_BUNDLE_REPO=YOU/hookseq-demo-bundle` (Settings, Variables).
5. **Pin the code**: tag the code repo (`git tag v0.1-demo && git push origin v0.1-demo`) and set
   `HOOKREC_REF` in `app.py` to that tag or a full commit SHA, never a branch. `app.py` installs
   `hookrec` with `pip --no-deps`, so the Space needs only the packages in `requirements.txt`
   (torch 2.8.0 is on ZeroGPU's supported list, 2.8.0 to 2.13.0 when checked; gradio is pinned to
   the tested 6.x, also as `sdk_version` in the README front-matter). Re-check the supported
   torch list at https://huggingface.co/docs/hub/spaces-zerogpu before bumping.
6. **Push the kit**: copy `deploy/hf_space/{app.py,requirements.txt,README.md}` to the Space repo
   and push.

Notes: the Space runs Python 3.12 (supported by ZeroGPU, and the version gradio 6.31.0 was tested
on). The engine never touches CUDA at import. On ZeroGPU it moves the models and item vectors to
`cuda` once at construction, as the ZeroGPU docs advise (CUDA is emulated outside `spaces.GPU` and
the transfer deferred), and the two `spaces.GPU` scoring functions only compute. Without
`spaces` it runs on CPU, or moves to CUDA on first use if available. Session items
outside the bundle's item subset are encoded as an "unknown" item; the bundle's sample users, and
every item of their windows, are always in the subset.
