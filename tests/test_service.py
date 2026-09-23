import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from reliability.bundle import load_bundle, save_bundle
from reliability.server import RecommendationApp, PROFILES, TITLES, demo_model, handler_factory


class BundleTests(unittest.TestCase):
    def test_static_demo_matches_python_model(self):
        bundle = json.loads((Path(__file__).resolve().parent.parent / "web" / "demo_bundle.json").read_text())
        model = demo_model()
        self.assertEqual(bundle["titles"], TITLES)
        self.assertEqual(bundle["profiles"], PROFILES)
        self.assertFalse(bundle["gate_open"])
        self.assertEqual(bundle["lifetime"], model.lifetime)
        self.assertEqual(bundle["recent"], model.recent)
        self.assertEqual(bundle["neighbors"], {item: [list(value) for value in entries]
                                               for item, entries in model.neighbors.items()})

    def test_reload_parity_and_integrity(self):
        model = demo_model()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle"
            save_bundle(model, path, "invented", "a" * 64)
            loaded, manifest = load_bundle(path)
            self.assertEqual(manifest["catalog_items"], len(model.lifetime))
            self.assertEqual(loaded.top_k(("atlas",), alpha=.75),
                             model.top_k(("atlas",), alpha=.75))
            with (path / "model.json.gz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_bundle(path)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = RecommendationApp(demo_model(), gate_open=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(self.app))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, payload):
        request = Request(self.base + "/api/recommend", json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_health_live_shadow_and_failure_fallback(self):
        with urlopen(self.base + "/api/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        with urlopen(self.base + "/app.js") as response:
            self.assertIn(b"staticRank", response.read())
        with urlopen(self.base + "/demo_bundle.json") as response:
            self.assertEqual(json.load(response)["schema"], 1)
        live = self.post({"history": ["atlas", "comet"], "k": 5})
        self.assertEqual(live["method"], "hybrid")
        self.assertEqual(live["active"], live["shadow"])
        self.assertNotIn("atlas", [item["id"] for item in live["active"]])
        failed = self.post({"history": ["atlas"], "simulate_failure": True})
        self.assertEqual(failed["method"], "recent")
        self.assertEqual(failed["active"], failed["baseline"])
        self.assertEqual(failed["fallback"], "challenger_unavailable")
        cold = self.post({"history": []})
        self.assertEqual(cold["fallback"], "cold_history")
        with urlopen(self.base + "/api/metrics") as response:
            self.assertEqual(json.load(response)["counts"]["requests"], 3)

    def test_gate_closed_and_invalid_request(self):
        self.app.gate_open = False
        closed = self.post({"history": ["atlas"]})
        self.assertEqual(closed["active"], closed["baseline"])
        self.assertEqual(closed["fallback"], "validation_gate_closed")
        with self.assertRaises(HTTPError) as caught:
            self.post({"history": "atlas"})
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()

    def test_neural_shadow_is_separate_from_active_route_and_can_fallback(self):
        class StubRetriever:
            def top_k(self, history, k):
                return ("orbit", "signal")[:k] if history else None

        self.app.neural = StubRetriever()
        shadowed = self.post({"history": ["atlas"], "k": 2})
        self.assertEqual([item["id"] for item in shadowed["neural_shadow"]], ["orbit", "signal"])
        self.assertEqual(shadowed["method"], "hybrid")
        self.app.neural_gate_open = True
        promoted = self.post({"history": ["atlas"], "k": 2})
        self.assertEqual(promoted["method"], "neural")
        self.assertEqual(promoted["active"], promoted["neural_shadow"])
        cold = self.post({"history": []})
        self.assertNotEqual(cold["method"], "neural")
        self.assertEqual(cold["neural_fallback"], "unsupported_history")


if __name__ == "__main__":
    unittest.main()
