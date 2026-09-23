# HTTP interface

`python -m reliability.server` serves a `ThreadingHTTPServer` bound to `127.0.0.1`. There
is no authentication, no cookie and no session: identity travels in the request body, as
the history you are willing to send. The server writes no access log — submitted histories
are never written anywhere — and the held-sentence list described under
`POST /api/utterance` lives in process memory only, cleared when the process stops.

Every response is JSON unless stated otherwise, with `Cache-Control: no-store` and
`X-Content-Type-Options: nosniff`. A `POST` body must be between 1 and 16,384 bytes.

## Starting the service

```powershell
python -m reliability.server --port 8899
python -m reliability.server --bundle runs/musical-bundle --aggregate runs/musical-replay/aggregate.json --port 8899
python -m reliability.server --bundle runs/musical-bundle --aggregate runs/musical-replay/aggregate.json --port 8899 --text data/text
```

| Flag | Effect |
| --- | --- |
| `--port` | Port to bind on `127.0.0.1`. Defaults to `8765`. |
| `--bundle`, `--aggregate` | Serve the real model and the real validation gate. Must be supplied together; the manifest, the evaluation archive and the bundle must agree on category and source hash. |
| `--neural-weights`, `--neural-aggregate`, `--neural-train` | Add the two-tower challenger as a shadow. Requires `--bundle`, and all four together. |
| `--text` | Directory of locally fetched product text, as written by `tools/text_corpus.py`. Enables `POST /api/utterance`. |
| `--embeddings` | Fold the encoded route into the phrase answer. Needs `--text`, and the `language` extra. |
| `--device` | `auto` (default), `cpu` or `cuda`; used only with `--embeddings`. |

Startup refuses `--embeddings needs --text` and `--text needs --bundle: invented items have
no product text`. A corpus that fails its own checks does not stop the service: the failure
is printed once, and `POST /api/utterance` answers with `phrase_fallback` set.

Without a bundle the service runs on nine invented titles, `data_scope` reads
`invented demo`, and nothing in a response may be read as a measurement.

## GET /api/health

`{"status": "ok", "data_scope": "...", "catalog_items": 22753, "product_text": "..."}`
(figure shown for the Musical Instruments research model).

`data_scope` is `invented demo` or `<category> local research model`. `product_text` says
what the wording route can read: `no product text loaded`, or a count of loaded product
descriptions and the directory they came from.

## POST /api/recommend

```json
{"history": ["atlas", "comet"], "k": 5, "simulate_failure": false}
```

`history` is an array of at most 200 strings of 1–128 characters; IDs the model does not
know are dropped rather than rejected. `k` is an integer from 1 to 20, default 5.
`simulate_failure` must be a boolean and stands the challenger down for one request.

The response separates the routes rather than blending them:

| Field | Meaning |
| --- | --- |
| `active` | What the service is serving now: challenger when the gate is open and it answers, baseline otherwise. Each row is `{"id", "title", "score"}`. |
| `baseline` | The selected fallback, always computed. |
| `shadow` | The challenger's own list, always computed, served or not. |
| `neural_shadow`, `neural_gate_open`, `neural_fallback` | The two-tower arm's list, its validation decision, and why it did not answer (`unsupported_history`, `neural_unavailable`). Absent when no neural weights were supplied. |
| `method` | `hybrid` when the challenger is being served, `neural` when the two-tower arm cleared its own gate, otherwise the baseline name (`recent` or `lifetime`). |
| `fallback` | `challenger_unavailable`, `validation_gate_closed`, `cold_history`, `unsupported_history`, or `null`. |
| `gate_open`, `alpha` | The persisted validation decision the service was started with. |
| `data_scope`, `local_rank_ms` | Whether this is a research model or the invented demo, and in-process ranking time only. |

## POST /api/utterance

```json
{"phrase": "a warm tube amp for a small room, not a pedalboard", "history": ["atlas"], "k": 5, "remember": true}
```

`phrase` is required, must be a non-empty string of at most 400 characters, and is parsed
into the terms to look for, the terms ruled out and a stated price ceiling. `history` and
`k` behave as above. `remember` must be a boolean and defaults to true. The route also
answers the behavioural question, so it is counted as a request as well as an utterance.

On top of every field of `POST /api/recommend`:

| Field | Meaning |
| --- | --- |
| `said` | The sentence, trimmed, as received. |
| `heard` | Terms the wording actually scored, after stopwords and negation handling. |
| `ruled_out` | Terms the sentence excluded. |
| `budget` | A stated price ceiling as a number, or `null`. Products with a recorded price above it are dropped from the wording's answer; products with no recorded price are kept, and if every candidate is known to be above the figure the figure is dropped rather than returning nothing. |
| `spoken` | What the wording selected — empty when it could not answer. |
| `combined` | What the person sees: the wording's list, or the behavioural one when the wording stood down. |
| `answering` | `spoken` or `behavioural`. |
| `phrase_fallback` | `text_corpus_unavailable`, `nothing_to_match`, `phrase_matched_nothing`, or `null`. |
| `product_text` | Same note as `/api/health`. |
| `phrase_ms` | Time spent on the wording route, in milliseconds. |
| `held` | How many sentences the service is currently holding. |

A sentence is held only when it actually answered, and only the most recent eight remain in
process memory. Nothing is written to disk or to a log.

## GET /api/ambient

Volunteers an answer from a sentence the service is already holding; the request carries no
wording. It increments the `offerings` counter.

With nothing held it returns `{"offered": false, "held": 0, "spoken": [], "combined": [],
"answering": "none", "fallback": "no_standing_sentence", "data_scope": "..."}`. Otherwise
it returns the `POST /api/utterance` fields, re-derived with `remember` off, plus
`"offered": true`, `offered_from_request` (the request number the sentence arrived on) and
`offered_after_requests` (how many requests later it was answered from).

## GET /api/metrics

`{"counts": {...}, "local_rank_ms": {"p50": ..., "p95": ...}, "scope": "in-process ranking
only; no network or deployment capacity claim"}`

The counters are `requests`, `hybrid_routes`, `neural_routes`, `baseline_routes`,
`fallbacks`, `invalid_requests`, `phrase_requests`, `phrase_answers`, `sentences_held` and
`offerings`. Latency covers ranking inside the process and excludes HTTP, startup,
concurrency and anything over a network.

## GET /api/profiles

`{"profiles": {"cosmic": [...], "story": [...], "mixed": [...], "new": []}, "titles": {...}}`
— the invented demo's starting points and its nine made-up titles, served only while
`data_scope` is `invented demo`.

## Errors

An empty or oversized body, unparseable JSON, a request body that is not an object, an
unknown field, a `k` outside 1 to 20, a malformed history, a bad `phrase` and a non-boolean
flag all return `400` with `{"error": "..."}` and increment `invalid_requests`. The messages
are the validation text in the source, for example `request body must be 1–16384 bytes`,
`invalid request fields`, `phrase must be a non-empty string`,
`phrase must be at most 400 characters` and
`history must be at most 200 item IDs; k must be 1–20`. Any other path returns `404` with
`{"error": "not found"}`; a method other than `GET` or `POST` gets the standard library's
`501`. The browser demo substitutes its own `service_unavailable` note when the request does
not reach the server at all; the service never sends that value.
