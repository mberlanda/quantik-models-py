"""`python -m quantik_models.play` — start the service and print where it is.

Defaults are chosen for the actual use: play from a phone on the home
WiFi, against whatever is staged, with the games kept somewhere they will
survive a `runs/` cleanup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import SERVICE_VERSION
from .server import lan_address, make_server, serve_forever
from .service import PlayService

# Not under `runs/`. These records are irreplaceable in a way checkpoints
# are not — a checkpoint can be retrained, a game someone played cannot be
# replayed — and `runs/` is gitignored and routinely deleted wholesale.
DEFAULT_DB = Path.home() / ".local" / "share" / "quantik" / "games.db"

# The visualizer, if it is checked out beside this repo. A sibling default
# rather than a required flag: that layout is what `quantik-ns` is. The
# repository root, not `src/` — `index.html` lives at the top and pulls
# `src/*.js`, so serving `src/` would serve the scripts and no page.
DEFAULT_STATIC = Path(__file__).resolve().parents[4] / "quantik-qfen-visualizer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantik-play", description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("staging"),
        help="directory of model subdirectories, each with manifest.json and "
        "weights.safetensors (default: staging)",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"game store (default: {DEFAULT_DB})")
    parser.add_argument(
        "--static",
        type=Path,
        default=DEFAULT_STATIC,
        help="directory to serve the browser app from",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: every interface)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-store", action="store_true", help="serve moves but record nothing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    service = PlayService(args.models)
    models = service.list_models()
    ready = [m for m in models if m["status"] == "ready"]

    db_path = None if args.no_store else args.db
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    static_dir = args.static if args.static.is_dir() else None

    print(f"quantik play service {SERVICE_VERSION}")
    print(f"  models     {args.models}  ({len(ready)} ready of {len(models)} found)")
    for model in models:
        # A refused model is printed with its reason rather than omitted.
        # Silence here is the failure mode: a model missing from the
        # dropdown looks like one that was never trained.
        mark = "  ok " if model["status"] == "ready" else "  -- "
        print(f"  {mark}{model['model_id']}" + (f"  ({model['reason']})" if model["reason"] else ""))
    print(f"  opponents  {len(service.list_opponents())}")
    print(f"  store      {db_path or 'disabled'}")
    print(f"  app        {static_dir or 'not served — pass --static'}")
    print()
    print(f"  local      http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  this WiFi  {lan_address(args.port)}")
    print()
    # stdout is block-buffered whenever it is not a terminal, so a banner
    # printed before `serve_forever` blocks appears only when the process
    # exits. The whole point of the banner is the address to type into a
    # phone *while* it runs.
    sys.stdout.flush()

    server = make_server(
        service, host=args.host, port=args.port, db_path=db_path, static_dir=static_dir
    )
    try:
        serve_forever(server)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
