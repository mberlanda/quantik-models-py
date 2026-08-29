"""The HTTP surface: five routes, the visualizer, and a LAN address.

`http.server` from the standard library, not a framework. The base
dependency of this package is numpy alone with torch in an extra, and that
posture is deliberate; a play service is not the place to spend it. The
work a framework would do here is small — five routes, JSON in and out, a
static directory — and the one thing it would genuinely buy, async
concurrency, buys nothing, because inference is serialized behind a lock
anyway (a shared evaluator and shared MCTS state).

What matters instead is that the server is *threading*: a 128-simulation
move takes a second or more, and a single-threaded server would make one
player's move freeze the board on every other device in the house.
`ThreadingHTTPServer` keeps the static files and the listing routes
answering while a move computes, and the lock inside `PlayService` is what
keeps the shared state safe underneath that.
"""

from __future__ import annotations

import json
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import record as rec
from . import store
from . import CONTRACT_VERSION, SERVICE_VERSION
from .service import PlayService, ServiceError

_MAX_BODY = 1 << 20  # 1 MiB: a finished Quantik game is a few hundred bytes.

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def lan_address(port: int) -> str:
    """The URL to type into a phone, resolved without assuming a platform.

    Opening a UDP socket toward a non-routable address makes the OS pick
    the interface it would actually use to leave this machine, and its
    local address is the one other devices can reach. Nothing is sent.
    `hostname -I` would be simpler and is Linux-only; `gethostname` often
    resolves to 127.0.0.1 on macOS, which is exactly the answer that looks
    right and does not work from another device.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 1))  # TEST-NET-1, guaranteed unroutable
        host = sock.getsockname()[0]
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    return f"http://{host}:{port}"


class PlayHandler(BaseHTTPRequestHandler):
    """Routes. Assigned `service`, `db_path` and `static_dir` by the factory."""

    service: PlayService
    db_path: Path | None
    static_dir: Path | None
    server_version = f"quantik-play/{SERVICE_VERSION}"
    protocol_version = "HTTP/1.1"

    # --- plumbing ------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs every request to stderr, which turns
        # a game into hundreds of lines. Errors still surface: they are
        # sent to the client and, for the unexpected ones, printed by
        # `_dispatch`.
        return

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The visualizer is served from this same origin, so CORS is moot
        # on the happy path. It is here for the case the plan anticipated:
        # the app opened from a file:// URL or another port, pointed at
        # this service by hand.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ServiceError(400, "a request body is required")
        if length > _MAX_BODY:
            raise ServiceError(413, f"request body exceeds {_MAX_BODY} bytes")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ServiceError(400, f"body is not valid JSON: {exc}") from exc

    def _dispatch(self, handler) -> None:
        try:
            status, payload = handler()
        except ServiceError as exc:
            self._send_json(exc.status, {"error": exc.message, "status": exc.status})
        except Exception as exc:  # noqa: BLE001 - the last line before a dropped connection
            # A traceback swallowed here is a request that dies silently
            # from the client's point of view; print it and answer.
            import traceback

            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}", "status": 500})
        else:
            self._send_json(status, payload)

    # --- routes --------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/opponents":
            self._dispatch(lambda: (200, {"opponents": self.service.list_opponents()}))
        elif path == "/api/models":
            self._dispatch(lambda: (200, {"models": self.service.list_models()}))
        elif path == "/api/games":
            self._dispatch(self._handle_summary)
        elif path.startswith("/api/"):
            self._send_json(404, {"error": f"no route {path}", "status": 404})
        else:
            self._serve_static(path)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/move/"):
            opponent_id = path[len("/api/move/") :]
            self._dispatch(lambda: self._handle_move(opponent_id))
        elif path == "/api/games":
            self._dispatch(self._handle_record)
        else:
            self._send_json(404, {"error": f"no route {path}", "status": 404})

    def _handle_move(self, opponent_id: str) -> tuple[int, Any]:
        from urllib.parse import unquote

        return 200, self.service.choose_move(unquote(opponent_id), self._read_json())

    def _handle_record(self) -> tuple[int, Any]:
        payload = rec.validate_payload(self._read_json())
        result = rec.replay(payload["initial_qfen"], payload["move_action_indices"])
        opponent = self.service.opponent(payload.get("opponent_id"))
        game, meta, positions = rec.rows_for(
            payload,
            result,
            contract_version=CONTRACT_VERSION,
            service_version=SERVICE_VERSION,
            opponent=opponent,
            client_user_agent=self.headers.get("User-Agent"),
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connection() as conn:
            inserted = store.record_game(conn, game, meta, positions)

        disagreements = rec.discrepancies(result, payload)
        return (
            201 if inserted else 200,
            {
                "game_id": game["game_id"],
                "recorded": inserted,
                "winner": result.winner,
                "plies": result.plies,
                "terminal_reason": result.terminal_reason,
                # Never empty-and-omitted: a caller has to be able to tell
                # "the two rule implementations agreed" from "this server
                # does not report disagreements".
                "discrepancies": disagreements,
            },
        )

    def _handle_summary(self) -> tuple[int, Any]:
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(self.path).query)
        player = query.get("player", [None])[0]
        with self._connection() as conn:
            return 200, {
                "games": store.game_count(conn),
                "head_to_head": store.head_to_head(conn, player),
            }

    def _connection(self):
        if self.db_path is None:
            raise ServiceError(503, "this service was started without a game store")
        # One connection per request rather than one shared: sqlite3
        # connections are not safe to move between threads, and this is a
        # ThreadingHTTPServer. Opening costs microseconds against a move
        # that costs a second.
        return store.connect(self.db_path)

    # --- static --------------------------------------------------------

    def _serve_static(self, path: str) -> None:
        if self.static_dir is None:
            self._send_json(404, {"error": "no static directory is configured", "status": 404})
            return

        relative = path.lstrip("/") or "index.html"
        try:
            target = (self.static_dir / relative).resolve()
            target.relative_to(self.static_dir.resolve())
        except (ValueError, OSError):
            # `..` walked out of the served tree. Refused rather than
            # normalised, because a served file outside the directory is
            # the whole class of bug this check exists for.
            self._send_json(403, {"error": "path escapes the static directory", "status": 403})
            return

        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self._send_json(404, {"error": f"no such file {relative}", "status": 404})
            return

        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        # The visualizer is edited while the server runs; a cached script
        # is a change that appears not to have happened.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def make_server(
    service: PlayService,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    db_path: Path | None = None,
    static_dir: Path | None = None,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundPlayHandler",
        (PlayHandler,),
        {
            "service": service,
            "db_path": Path(db_path) if db_path else None,
            "static_dir": Path(static_dir).resolve() if static_dir else None,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_forever(server: ThreadingHTTPServer) -> None:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    thread.join()
