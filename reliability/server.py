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
MAX_PHRASE_CHARS = 400
HELD_SENTENCES = 8


def language_route(model: Recommender, directory: Path, category: str, *,
                   use_embeddings: bool = False, device: str = "auto",
                   baseline: str = "recent", alpha: float = .75, weight: float = 1.0,
                   repulsion: float = .5, price_penalty: float = .5,
                   enforce_ceiling: bool = True):
    """A ranker that answers in words, plus the real product names behind it.

    An interactive box is phrase-first: the sentence chooses, the behavioural
    scores break the tie. That is a different question from the measured arm,
    which asks whether ambient wording is worth more than a feed and therefore
    confines the phrase to the behavioural shortlist. Here the phrase may reach
    the whole shelf, because someone who asks for something specific expects to
    be answered rather than nudged. What the sentence rules out is subtracted at
    `repulsion`, so "not a pedal" costs a pedal, and a figure the person names is
    treated as a limit rather than a hint: a product whose recorded price is above it
    is not offered, while products with no recorded price stay eligible because the
    archive is silent about them rather than contrary.
    """
    from .language import TextRanker, build_index, load_corpus

    catalogue = set(model.lifetime)
    corpus, manifest = load_corpus(Path(directory), category, catalogue=catalogue)
    index, _ = build_index(corpus, catalogue=catalogue)
    vectors = None
    if use_embeddings:
        from .embeddings import load_embeddings

        vectors, _ = load_embeddings(Path(directory), category,
                                     expected_corpus_sha256=str(manifest.get("corpus_sha256")),
                                     device=device)
    ranker = TextRanker(model, index, alpha=alpha, baseline=baseline, weight=weight,
                        repulsion=repulsion, price_penalty=price_penalty, embeddings=vectors,
                        enforce_ceiling=enforce_ceiling)
    titles = {item: str(record.get("title") or item) for item, record in corpus.items()}
    return ranker, titles



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
                 label: str = "invented demo", neural=None, neural_gate_open: bool = False,
                 language=None, language_note: str = "no product text loaded"):
        if baseline not in ("lifetime", "recent") or not 0 <= alpha <= 1:
            raise ValueError("invalid serving decision")
        self.model = model
        self.baseline = baseline
        self.alpha = alpha
        self.gate_open = gate_open
        self.titles = titles or {}
        self.label = label
        self.neural = neural
        self.neural_gate_open = neural_gate_open
        self.language = language
        self.language_note = language_note
        self.standing: list[dict] = []
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
        neural_shadow = ()
        neural_fallback = None
        fallback = None
        try:
            if simulate_failure:
                raise RuntimeError("demonstrated challenger failure")
            shadow = self.model.top_k(history, self.baseline, self.alpha, k)
        except Exception:
            fallback = "challenger_unavailable"
        if self.neural is not None:
            try:
                if simulate_failure:
                    raise RuntimeError("demonstrated challenger failure")
                neural_result = self.neural.top_k(history, k)
                if neural_result is None:
                    neural_fallback = "unsupported_history"
                else:
                    neural_shadow = neural_result
            except Exception:
                neural_fallback = "neural_unavailable"
        if not self.gate_open:
            fallback = fallback or "validation_gate_closed"
        elif not history:
            fallback = fallback or "cold_history"
        elif not any(item in self.model.neighbors for item in history):
            fallback = fallback or "unsupported_history"
        method = "hybrid" if self.gate_open and history and fallback is None else self.baseline
        active_items = shadow if method == "hybrid" else baseline_items
        if self.neural_gate_open and neural_shadow and neural_fallback is None:
            method, active_items = "neural", neural_shadow
            fallback = None
        elapsed_ms = (perf_counter() - started) * 1000
        with self.lock:
            self.counts["requests"] += 1
            route_count = "neural_routes" if method == "neural" else "hybrid_routes" if method == "hybrid" else "baseline_routes"
            self.counts[route_count] += 1
            if fallback or neural_fallback:
                self.counts["fallbacks"] += 1
            self.latencies.append(elapsed_ms)
        decorate = self.decorate

        return {"active": decorate(active_items), "baseline": decorate(baseline_items),
                "shadow": decorate(shadow), "neural_shadow": decorate(neural_shadow),
                "neural_gate_open": self.neural_gate_open, "neural_fallback": neural_fallback,
                "method": method, "fallback": fallback,
                "gate_open": self.gate_open, "alpha": self.alpha,
                "data_scope": self.label, "local_rank_ms": round(elapsed_ms, 3)}

    def decorate(self, items):
        return [{"id": item, "title": self.titles.get(item, item)} for item in items]

    def hear(self, phrase: str, history: list[str] | None = None, k: int = 5,
             remember: bool = True):
        """Answer something the person said, and show what the behavioural route would do.

        The words are parsed into what to look for, what to rule out, and a budget.
        Whatever the phrase cannot answer, the validated behavioural route still does,
        and both are reported side by side rather than merged into one number.
        """
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("phrase must be a non-empty string")
        if len(phrase) > MAX_PHRASE_CHARS:
            raise ValueError(f"phrase must be at most {MAX_PHRASE_CHARS} characters")
        result = self.recommend(history or [], k)
        from .language import parse_utterance

        utterance = parse_utterance(phrase)
        started = perf_counter()
        spoken: tuple[str, ...] = ()
        note = None
        if self.language is None:
            note = "text_corpus_unavailable"
        elif utterance.empty:
            note = "nothing_to_match"
        else:
            prepared = self.language.prepare(tuple(history or ()))
            attraction, repulsion = self.language.route(utterance, prepared)
            spoken = self.language.rank_from(prepared, attraction, repulsion,
                                             utterance.price_ceiling, k)
            if not spoken:
                note = "phrase_matched_nothing"
        phrase_ms = (perf_counter() - started) * 1000
        result.update({"said": phrase.strip(), "heard": list(utterance.positive),
                       "ruled_out": list(utterance.negative),
                       "budget": utterance.price_ceiling,
                       "spoken": self.decorate(spoken),
                       "combined": self.decorate(spoken or [row["id"] for row in result["active"]]),
                       "answering": "spoken" if spoken else "behavioural",
                       "phrase_fallback": note, "product_text": self.language_note,
                       "phrase_ms": round(phrase_ms, 3)})
        with self.lock:
            self.counts["phrase_requests"] += 1
            if note is None:
                self.counts["phrase_answers"] += 1
            if remember and note is None:
                self.standing.append({"phrase": phrase.strip(), "history": list(history or ()),
                                      "k": k, "at_request": self.counts["requests"]})
                del self.standing[:-HELD_SENTENCES]
                self.counts["sentences_held"] += 1
            result["held"] = len(self.standing)
        return result

    def offer(self, k: int = 5):
        """Volunteer a product list from a sentence the person already left with us.

        Nothing new is submitted here: the request carries no wording, and the
        answer comes from a sentence held from earlier in the same session.
        """
        with self.lock:
            held = dict(self.standing[-1]) if self.standing else None
            request_number = self.counts["requests"]
        if held is None:
            with self.lock:
                self.counts["offerings"] += 1
            return {"offered": False, "held": 0, "spoken": [], "combined": [],
                    "answering": "none", "fallback": "no_standing_sentence",
                    "data_scope": self.label}
        answer = self.hear(held["phrase"], held["history"], held.get("k", k), remember=False)
        answer.update({"offered": True, "offered_from_request": held["at_request"],
                       "offered_after_requests": request_number - held["at_request"]})
        with self.lock:
            self.counts["offerings"] += 1
        return answer

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
            if path in ("/app.js", "/demo_bundle.json"):
                file = Path(__file__).resolve().parent.parent / "web" / path.removeprefix("/")
                content_type = "application/javascript; charset=utf-8" if path.endswith(".js") else "application/json"
                return self.respond(200, file.read_bytes(), content_type)
            if path == "/api/health":
                return self.respond(200, {"status": "ok", "data_scope": app.label,
                                          "catalog_items": len(app.model.lifetime),
                                          "product_text": app.language_note})
            if path == "/api/ambient":
                return self.respond(200, app.offer())
            if path == "/api/metrics":
                return self.respond(200, app.metrics())
            if path == "/api/profiles" and app.label == "invented demo":
                return self.respond(200, {"profiles": PROFILES, "titles": TITLES})
            return self.respond(404, {"error": "not found"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/recommend", "/api/utterance"):
                return self.respond(404, {"error": "not found"})
            fields = ({"history", "k", "simulate_failure"} if path == "/api/recommend"
                      else {"phrase", "history", "k", "remember"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16_384:
                    raise ValueError("request body must be 1–16384 bytes")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or set(request) - fields:
                    raise ValueError("invalid request fields")
                if path == "/api/recommend":
                    if not isinstance(request.get("simulate_failure", False), bool):
                        raise ValueError("simulate_failure must be boolean")
                    result = app.recommend(request.get("history", []), request.get("k", 5),
                                           request.get("simulate_failure", False))
                else:
                    if not isinstance(request.get("remember", True), bool):
                        raise ValueError("remember must be boolean")
                    result = app.hear(request.get("phrase", ""), request.get("history", []),
                                      request.get("k", 5), request.get("remember", True))
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
    parser.add_argument("--neural-weights", type=Path)
    parser.add_argument("--neural-aggregate", type=Path)
    parser.add_argument("--neural-train", type=Path)
    parser.add_argument("--text", type=Path, help="directory of locally fetched product text")
    parser.add_argument("--embeddings", action="store_true",
                        help="add the sentence-embedding route to the phrase answer")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if bool(args.bundle) != bool(args.aggregate):
        parser.error("--bundle and --aggregate must be supplied together")
    if any((args.neural_weights, args.neural_aggregate, args.neural_train)) and not all(
            (args.neural_weights, args.neural_aggregate, args.neural_train, args.bundle)):
        parser.error("neural weights, aggregate, train archive, and base bundle must be supplied together")
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
        if args.neural_weights:
            from .neural import load_retriever
            neural, neural_manifest = load_retriever(args.neural_weights, args.neural_train)
            neural_aggregate = json.loads(args.neural_aggregate.read_text(encoding="utf-8"))
            neural_decision = neural_aggregate["validation_decision"]
            if (neural_decision["category"] != manifest["category"] or
                    neural_decision["source_sha256"]["train"] != neural_manifest["train_sha256"] or
                    neural_aggregate["source_sha256"]["train"] != neural_manifest["train_sha256"] or
                    neural_decision["weights_sha256"] != neural_manifest["weights_sha256"]):
                raise ValueError("neural model and evaluation source differ")
            scores = model.recent if neural_decision["baseline_method"] == "recent" else model.lifetime
            neural.configure_blend(scores, neural_decision["blend_alpha"])
            app.neural = neural
            app.neural_gate_open = bool(neural_decision["gate_open"])
    else:
        app = RecommendationApp(demo_model(), titles=TITLES)
    if args.embeddings and not args.text:
        parser.error("--embeddings needs --text")
    if args.text:
        if not args.bundle:
            parser.error("--text needs --bundle: invented items have no product text")
        started = perf_counter()
        try:
            ranker, titles = language_route(app.model, args.text, manifest["category"],
                                            use_embeddings=args.embeddings, device=args.device,
                                            baseline=app.baseline, alpha=app.alpha)
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"phrase answers unavailable, behavioural route still serves: {error}",
                  flush=True)
        else:
            app.language, app.titles = ranker, titles
            app.language_note = f"{len(titles)} product descriptions under {args.text}"
            print(f"phrase answers ready: {app.language_note} in "
                  f"{(perf_counter() - started):.1f}s", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(app)) as server:
        print(f"Local demo: http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
