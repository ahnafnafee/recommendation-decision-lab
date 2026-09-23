"""Loopback-only recommendation service with a transparent validation gate."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

from .bundle import load_bundle
from .core import Recommender


TITLES = {
    "harbor": "Hollow Harbor", "atlas": "Atlas Drift", "comet": "Comet Circuit",
    "grove": "The Glass Grove", "lantern": "Lantern Vale", "tide": "Tide and Timber",
    "rune": "Runebound", "signal": "Signal Runner", "orbit": "Orbit of Ash",
}
PROFILES = {
    "cosmic": ["atlas", "comet"],
    "story": ["grove", "lantern"],
    "mixed": ["harbor", "signal"],
    "new": [],
}


def demo_model():
    groups = [
        ("atlas", "comet", "orbit"), ("atlas", "comet", "signal"),
        ("grove", "lantern", "tide"), ("grove", "lantern", "rune"),
        ("harbor", "signal", "atlas"), ("harbor", "tide", "grove"),
    ]
    events = []
    for group_index, group in enumerate(groups):
        for copy in range(4):
            for position, item in enumerate(group):
                events.append((f"invented-{group_index}-{copy}", item, 100 + position))
    return Recommender.train(events, TITLES, 1_000)


class RecommendationApp:
    def __init__(self, model: Recommender, baseline: str = "recent", alpha: float = .75,
                 gate_open: bool = False, titles: dict[str, str] | None = None,
                 label: str = "invented demo"):
        if baseline not in ("lifetime", "recent") or not 0 <= alpha <= 1:
            raise ValueError("invalid serving decision")
        self.model = model
        self.baseline = baseline
        self.alpha = alpha
        self.gate_open = gate_open
        self.titles = titles or {}
        self.label = label
        self.counts = Counter()
        self.latencies = deque(maxlen=1000)
        self.lock = Lock()

    def recommend(self, history: list[str], k: int = 5, simulate_failure: bool = False):
        if (not isinstance(history, list) or len(history) > 200 or
                any(not isinstance(item, str) or not item or len(item) > 128 for item in history)
                or not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20):
            raise ValueError("history must be at most 200 item IDs; k must be 1–20")
        started = perf_counter()
        baseline_items = self.model.top_k(history, self.baseline, 0.0, k)
        shadow = ()
        fallback = None
        try:
            if simulate_failure:
                raise RuntimeError("demonstrated challenger failure")
            shadow = self.model.top_k(history, self.baseline, self.alpha, k)
        except Exception:
            fallback = "challenger_unavailable"
        if not self.gate_open:
            fallback = fallback or "validation_gate_closed"
        elif not history:
            fallback = fallback or "cold_history"
        elif not any(item in self.model.neighbors for item in history):
            fallback = fallback or "unsupported_history"
        method = "hybrid" if self.gate_open and history and fallback is None else self.baseline
        active_items = shadow if method == "hybrid" else baseline_items
        elapsed_ms = (perf_counter() - started) * 1000
        with self.lock:
            self.counts["requests"] += 1
            self.counts["hybrid_routes" if method == "hybrid" else "baseline_routes"] += 1
            if fallback:
                self.counts["fallbacks"] += 1
            self.latencies.append(elapsed_ms)
        decorate = lambda items: [{"id": item, "title": self.titles.get(item, item)} for item in items]
        return {"active": decorate(active_items), "baseline": decorate(baseline_items),
                "shadow": decorate(shadow), "method": method, "fallback": fallback,
                "gate_open": self.gate_open, "alpha": self.alpha,
                "data_scope": self.label, "local_rank_ms": round(elapsed_ms, 3)}

    def metrics(self):
        with self.lock:
            latencies = sorted(self.latencies)
            counts = dict(self.counts)
        percentile = lambda q: round(latencies[int((len(latencies)-1)*q)], 3) if latencies else None
        return {"counts": counts, "local_rank_ms": {"p50": percentile(.5), "p95": percentile(.95)},
                "scope": "in-process ranking only; no network or deployment capacity claim"}


def handler_factory(app: RecommendationApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return  # Never log submitted histories.

        def respond(self, status, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                page = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_bytes()
                return self.respond(200, page, "text/html; charset=utf-8")
            if path == "/api/health":
                return self.respond(200, {"status": "ok", "data_scope": app.label,
                                          "catalog_items": len(app.model.lifetime)})
            if path == "/api/metrics":
                return self.respond(200, app.metrics())
            if path == "/api/profiles" and app.label == "invented demo":
                return self.respond(200, {"profiles": PROFILES, "titles": TITLES})
            return self.respond(404, {"error": "not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/recommend":
                return self.respond(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16_384:
                    raise ValueError("request body must be 1–16384 bytes")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or set(request) - {"history", "k", "simulate_failure"}:
                    raise ValueError("invalid request fields")
                if not isinstance(request.get("simulate_failure", False), bool):
                    raise ValueError("simulate_failure must be boolean")
                result = app.recommend(request.get("history", []), request.get("k", 5),
                                       request.get("simulate_failure", False))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                with app.lock:
                    app.counts["invalid_requests"] += 1
                return self.respond(400, {"error": str(error)})
            return self.respond(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if bool(args.bundle) != bool(args.aggregate):
        parser.error("--bundle and --aggregate must be supplied together")
    if args.bundle:
        model, manifest = load_bundle(args.bundle)
        aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
        if not aggregate["experiment"].startswith(manifest["category"] + " "):
            raise ValueError("bundle and evaluation category differ")
        decision = aggregate["validation_decision"]
        if decision["source_sha256"]["train"] != manifest["source_sha256"]:
            raise ValueError("model source and evaluation source differ")
        app = RecommendationApp(model, decision["baseline"], decision["challenger_alpha"],
                                decision["gate_open"], label=manifest["category"] + " local research model")
    else:
        app = RecommendationApp(demo_model())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(app)) as server:
        print(f"Local demo: http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
