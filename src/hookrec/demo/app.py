"""Gradio app: session -> next tracks (HookSeq vs SASRec) and new-track audience.

python -m hookrec.demo.app --bundle DIR [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import logging
import math

from hookrec.demo.engine import EVENT_TOKEN, DemoEngine

SESSION_HEADERS = ["track id", "event", "organic"]
REC_HEADERS = ["rank", "track id", "artist id", "popularity", "score"]
AUDIENCE_HEADERS = ["rank", "user id", "score"]
EVENTS = sorted(EVENT_TOKEN)


def parse_session(rows) -> list[tuple[int, str, bool]]:
    """Editor rows [[track id, event, organic], ...] -> engine session. Blank rows are skipped."""
    session = []
    for n, row in enumerate(rows if rows is not None else [], start=1):
        item, event, organic = (list(row) + [None] * 3)[:3]
        if item is None or (isinstance(item, float) and math.isnan(item)) or item == "":
            continue
        try:
            item_id = int(float(item))
        except (TypeError, ValueError):
            raise ValueError(f"row {n}: track id {item!r} is not a number") from None
        event = str(event).strip() if event is not None else ""
        if event not in EVENT_TOKEN:
            raise ValueError(f"row {n}: event must be one of {EVENTS}, got {event!r}")
        session.append((item_id, event, str(organic).strip().lower() in ("true", "1", "yes")))
    return session


def _rec_rows(recs: list[dict]) -> list[list]:
    return [
        [r, d["item_id"], d["artist_id"], d["popularity"], round(d["score"], 4)]
        for r, d in enumerate(recs, start=1)
    ]


def recommend_both(engine: DemoEngine, rows, k):
    """(HookSeq rows, SASRec rows, status) for the editor contents; errors go to the status."""
    try:
        session = parse_session(rows)
        out = [_rec_rows(engine.recommend(session, m, int(k))) for m in ("hookseq", "sasrec")]
    except ValueError as e:
        return [], [], f"**Error:** {e}"
    known = engine.to_compact([s[0] for s in session]) != engine.unknown_index
    msg = f"{len(session)} events"
    if not known.all():
        msg += f", {int((~known).sum())} outside the bundle's item subset (treated as unknown)"
    return out[0], out[1], msg


def load_sample(engine: DemoEngine, i):
    rows = engine.sample_user_session(int(i))
    return [[it, ev, org] for it, ev, org in rows]


def audience(engine: DemoEngine, track, k):
    if track is None:
        return [], "**Error:** pick a track"
    try:
        rows = engine.new_track_audience(int(track), int(k))
    except ValueError as e:
        return [], f"**Error:** {e}"
    return [[r, d["uid"], round(d["score"], 4)] for r, d in enumerate(rows, start=1)], (
        f"{len(rows)} of {engine.n_sample_users} sample users"
    )


def build_app(engine: DemoEngine):
    import gradio as gr

    with gr.Blocks(title="HookSeq demo") as demo:
        gr.Markdown(
            "# HookSeq demo\nNext-track recommendations from a listening session, HookSeq "
            "(audio, artist, event and organic signals) against a plain SASRec."
        )
        with gr.Tab("Session → next tracks"):
            gr.Markdown(
                "Edit the session (oldest event first; add rows at the bottom) or load a sample "
                "user. Events: " + ", ".join(f"`{e}`" for e in EVENTS) + ". Toggle `organic` to "
                "see HookSeq react to it."
            )
            with gr.Row():
                sample = gr.Slider(
                    0, max(engine.n_sample_users - 1, 0), value=0, step=1, label="Sample user"
                )
                load = gr.Button("Load sample user")
                k = gr.Slider(1, 50, value=10, step=1, label="Top k")
            session = gr.Dataframe(
                value=load_sample(engine, 0),
                headers=SESSION_HEADERS,
                datatype=["number", "str", "bool"],
                column_count=3,
                interactive=True,
                type="array",
                label="Session",
            )
            run = gr.Button("Recommend", variant="primary")
            status = gr.Markdown()
            with gr.Row():
                hs_out = gr.Dataframe(headers=REC_HEADERS, label="HookSeq", interactive=False)
                sas_out = gr.Dataframe(headers=REC_HEADERS, label="SASRec", interactive=False)
            load.click(lambda i: load_sample(engine, i), sample, session)
            run.click(
                lambda r, kk: recommend_both(engine, r, kk), [session, k], [hs_out, sas_out, status]
            )
        with gr.Tab("New track audience"):
            gr.Markdown(
                "Tracks with audio but no ID embedding (never seen enough in training). HookSeq "
                "ranks the sample users most likely to take to the track."
            )
            choices = [
                (f"{t} (artist {engine.artist[i]})", t)
                for i, t in enumerate(engine.raw_ids.tolist())
                if not engine.has_id[i] and engine.has_audio[i]
            ]
            track = gr.Dropdown(choices, value=choices[0][1] if choices else None, label="Track")
            ka = gr.Slider(1, 50, value=10, step=1, label="Top k users")
            go = gr.Button("Find audience", variant="primary")
            a_status = gr.Markdown()
            a_out = gr.Dataframe(headers=AUDIENCE_HEADERS, label="Audience", interactive=False)
            go.click(lambda t, kk: audience(engine, t, kk), [track, ka], [a_out, a_status])
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="demo bundle directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build_app(DemoEngine(args.bundle)).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
