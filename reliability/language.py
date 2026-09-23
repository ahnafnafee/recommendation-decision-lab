"""Phrase-conditioned ranking: an utterance becomes a sparse score adjustment.

The module keeps the repository's exactness rule. A spoken or written phrase is
turned into positive and negative terms, those terms are matched against a local
product-text corpus, and the resulting adjustment is added to the committed
score under a frozen weight. With the weight at zero the ranking reproduces the
committed hybrid exactly, and with an empty phrase it does too.

Text scoring uses NumPy when present and a dictionary path otherwise, so the
standard-library build keeps working. Fitted artefacts are written to ignored
directories; aggregate JSON holds no identifiers or phrases.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import heapq
import json
from math import log
from pathlib import Path
import re

try:
    from numpy import add as _np_add, float32 as _np_float32, zeros as _np_zeros
except ImportError:  # Standard-library build keeps working without NumPy.
    _np_add = _np_float32 = _np_zeros = None

from .core import Recommender, last_distinct

CORPUS_SCHEMA = 3

WORD = re.compile(r"[a-z0-9][a-z0-9'+\-]*")
PRICE = re.compile(r"(?:under|below|less than|up to|max(?:imum)?|cheaper than|<)\s*\$?\s*"
                   r"(\d{1,4}(?:,\d{3})*(?:\.\d{1,2})?)")
NEGATION_CUES = frozenset({"not", "no", "never", "without", "neither", "nor", "avoid", "avoiding",
                           "exclude", "excluding", "dislike", "dislikes", "disliked", "hate", "hates",
                           "hated", "remove", "removes", "minus", "cant", "cannot", "wont", "dont",
                           "doesnt", "isnt", "wasnt", "nothing", "none", "lack", "lacks"})
AFFIRMATIVE_RESTART = frozenset({"but", "except", "though", "although", "while", "just", "only", "prefer"})
# "rather than X" and "instead of X" rule X out; a bare "rather" is filler.
COMPARATIVE_CUES = {"rather": "than", "instead": "of"}
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have
he her his i if in is it its me my no not of on or our she so that the their them then there these they
this to too us was we were what when where which who will with would you your""".split())
DOCUMENT_FIELDS = ("title", "brand", "categories", "features", "description", "details")


@dataclass(frozen=True)
class Utterance:
    """Terms a phrase asks for, terms it rules out, and an optional price ceiling."""

    positive: tuple[str, ...]
    negative: tuple[str, ...]
    price_ceiling: float | None
    raw: str = field(default="", repr=False, compare=False)

    @property
    def empty(self) -> bool:
        return not self.positive and not self.negative and self.price_ceiling is None


def normalised(text: str) -> str:
    """Lower-case with apostrophes folded away, so "don't" and "dont" agree."""
    return (text or "").lower().replace("\u2019", "'").replace("'", "")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(word for word in WORD.findall(normalised(text))
                 if word not in STOPWORDS and len(word) > 1)


def price_ceiling(text: str) -> float | None:
    """A stated ceiling: "under $1,200", "up to 300", "< 50"."""
    match = PRICE.search(normalised(text))
    return float(match.group(1).replace(",", "")) if match else None


def parse_utterance(text: str) -> Utterance:
    """Split a phrase into what is wanted and what is ruled out.

    Negation is scoped to the words after the cue and ends at the next cue, at a
    conjunction that restarts the request ("but", "only"), or at end of text.
    "rather than X" and "instead of X" rule X out; a bare "rather" is filler.
    """
    words = [word for word in WORD.findall(normalised(text)) if len(word) > 1]
    positive: list[str] = []
    negative: list[str] = []
    blocked = False
    index = 0
    while index < len(words):
        word = words[index]
        follower = words[index + 1] if index + 1 < len(words) else None
        cue = COMPARATIVE_CUES.get(word)
        if cue is not None and follower == cue:
            blocked, index = True, index + 2
            continue
        if word in NEGATION_CUES:
            blocked, index = True, index + 1
            continue
        if word in AFFIRMATIVE_RESTART:
            blocked, index = False, index + 1
            continue
        if word not in STOPWORDS:
            (negative if blocked else positive).append(word)
        index += 1
    return Utterance(tuple(dict.fromkeys(positive)), tuple(dict.fromkeys(negative)),
                     price_ceiling(text), text or "")


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {value[key]}" for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return " ; ".join(_as_text(entry) for entry in value)
    return "" if value is None else str(value)


def document_text(record: dict, review_limit: int = 0, description_chars: int = 400) -> str:
    """One searchable string per item. Field order is fixed, so hashing is stable."""
    parts = [_as_text(record.get("title")), _as_text(record.get("brand"))]
    categories = record.get("categories")
    if isinstance(categories, list) and categories and isinstance(categories[0], list):
        categories = [entry for branch in categories for entry in branch]
    parts.append(_as_text(categories))
    parts.append(_as_text(record.get("features")))
    description = _as_text(record.get("description"))[:description_chars]
    parts.append(description)
    if review_limit:
        parts.extend(_as_text(review)[:200] for review in (record.get("review_prose") or ())[:review_limit])
    details = record.get("details")
    if details and not record.get("features"):
        parts.append(_as_text(details)[:description_chars])
    return " ".join(part for part in parts if part)


def load_corpus(directory: Path, category: str, catalogue: set[str] | None = None,
                review_limit: int = 0):
    """Read the local corpus plus its manifest; refuse a corpus from another cut."""
    manifest_path = directory / f"{category}.corpus_manifest.json"
    corpus_path = directory / f"{category}.corpus.jsonl.gz"
    if not manifest_path.exists() or not corpus_path.exists():
        raise ValueError(f"no product text for {category} under {directory}; "
                         f"build it with tools/text_corpus.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("category") != category:
        raise ValueError("corpus category mismatch")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"corpus manifest schema {manifest.get('schema')!r} is not {CORPUS_SCHEMA}")
    records = {}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            records[record["asin"]] = record
    if catalogue is not None:
        coverage = 1 - len(catalogue - set(records)) / max(1, len(catalogue))
        if coverage < 0.5:
            raise ValueError(f"corpus covers only {coverage:.1%} of the catalogue")
    return records, manifest


def load_voices(directory: Path, category: str, before_ms: int | None = None):
    """What people wrote themselves, newest first, optionally cut at a timestamp."""
    path = directory / f"{category}.voices.jsonl.gz"
    if not path.exists():
        return {}
    bank = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            entries = [entry for entry in record["entries"]
                       if before_ms is None or entry[0] < before_ms]
            if entries:
                bank[record["user_id"]] = entries
    return bank


def own_words(bank: dict, user_id: str, exclude_item: str | None = None,
              limit: int = 3) -> str:
    """The most recent thing this person wrote, as the phrase they are answering."""
    entries = [entry for entry in bank.get(user_id, ()) if entry[1] != exclude_item][:limit]
    return " ".join(entry[3] for entry in entries)


def load_queries(directory: Path, category: str, catalogue: set[str] | None = None):
    """Published phrase-to-item pairs, or an empty list when they were not fetched.

    Each entry is (identifier, phrase, item, source). The phrases were written by
    other people about other products, so their shape is independent of anything
    this lab generates.
    """
    path = directory / f"{category}.queries.jsonl.gz"
    if not path.exists():
        return []
    pairs = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if catalogue is None or record["asin"] in catalogue:
                pairs.append((record["query_id"], record["text"], record["asin"], record["source"]))
    return pairs


class LexicalIndex:
    """BM25 over the product corpus, with exact full-catalogue scoring arrays."""

    def __init__(self, documents: dict[str, str], k1: float = 1.2, b: float = 0.75,
                 prices: dict[str, float] | None = None):
        self.items = sorted(documents)
        self.position = {item: index for index, item in enumerate(self.items)}
        self.k1, self.b = k1, b
        self.prices = dict(prices or {})
        self.doc_length = [0] * len(self.items)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, item in enumerate(self.items):
            counts = Counter(tokenize(documents[item]))
            self.doc_length[index] = sum(counts.values())
            for term, count in counts.items():
                postings[term][index] = count
        self.postings = {term: dict(entries) for term, entries in postings.items()}
        total = sum(self.doc_length)
        self.average_length = total / max(1, len(self.items))
        self.document_count = len(self.items)
        self.idf = {term: log(1 + (self.document_count - len(entries) + 0.5) / (len(entries) + 0.5))
                    for term, entries in self.postings.items()}
        self._cached: dict[str, tuple] = {}

    def __len__(self) -> int:
        return self.document_count

    def _weights(self, term: str):
        cached = self._cached.get(term)
        if cached is None:
            entries = self.postings.get(term, {})
            pairs = sorted(entries.items())
            docs = [index for index, _ in pairs]
            idf = self.idf.get(term, 0.0)
            values = [idf * (count * (self.k1 + 1) /
                             (count + self.k1 * (1 - self.b + self.b * self.doc_length[index] /
                                                 self.average_length)))
                      for index, count in pairs]
            if _np_float32 is not None:
                packed = (_np_zeros(self.document_count, dtype=_np_float32),)
                packed[0][docs] = values
            else:
                packed = ({doc: value for doc, value in zip(docs, values)},)
            cached = self._cached[term] = packed
        return cached[0]

    def score(self, terms) -> dict[str, float] | None:
        """Sparse term scores over the whole catalogue, or None when nothing matches."""
        terms = [term for term in dict.fromkeys(terms) if term in self.postings]
        if not terms:
            return None
        if _np_float32 is not None:
            total = _np_zeros(self.document_count, dtype=_np_float32)
            for term in terms:
                _np_add(total, self._weights(term), out=total)
            return {self.items[index]: float(value)
                    for index, value in enumerate(total) if value > 0}
        total: dict[int, float] = defaultdict(float)
        for term in terms:
            for doc, value in self.postings_score_dict(term).items():
                total[doc] += value
        return {self.items[doc]: value for doc, value in total.items() if value > 0}

    def postings_score_dict(self, term: str) -> dict[int, float]:
        idf = self.idf.get(term, 0.0)
        return {index: idf * (count * (self.k1 + 1) /
                              (count + self.k1 * (1 - self.b + self.b * self.doc_length[index] /
                                                  self.average_length)))
                for index, count in self.postings.get(term, {}).items()}

    def price_of(self, item: str) -> float | None:
        price = self.prices.get(item)
        return float(price) if isinstance(price, (int, float)) else None

    def term_document_frequency(self, token: str) -> int:
        """How many products mention a word: the line between a brand and a category."""
        return len(self.postings.get(token, ()))

    def coverage(self, terms) -> float:
        """Fraction of catalogue documents containing at least one term."""
        hit = set()
        for term in terms:
            hit.update(self.postings.get(term, ()))
        return len(hit) / max(1, self.document_count)

    def statistics(self) -> dict:
        lengths = sorted(self.doc_length)
        return {"documents": self.document_count,
                "distinct_terms": len(self.postings),
                "document_length_median": lengths[len(lengths) // 2] if lengths else 0,
                "documents_with_zero_text": sum(1 for length in self.doc_length if not length)}


def build_index(records: dict[str, dict], catalogue: set[str] | None = None,
                review_limit: int = 4, **options) -> tuple[LexicalIndex, dict]:
    """Index every catalogue item; items with no text at all stay in, scored zero."""
    wanted = sorted(catalogue) if catalogue is not None else sorted(records)
    documents = {item: document_text(records.get(item, {}), review_limit) for item in wanted}
    prices = {item: records[item]["price"] for item in wanted
              if isinstance(records.get(item, {}).get("price"), (int, float))}
    index = LexicalIndex(documents, prices=prices, **options)
    return index, {"indexed": len(documents), "without_any_text":
                   sum(1 for text in documents.values() if not text), "with_price": len(prices),
                   **index.statistics()}


def rank_normalised(scores: dict[str, float]) -> dict[str, float]:
    """Map unbounded retrieval scores onto the 0–1 range the blend expects.

    Distinct scores keep their order and equal scores share one value, so two
    items that match equally are not pulled apart by their identifiers. The best
    score maps to 1.0 and the worst to 0.0; items with no retrieval score are
    absent and contribute nothing.
    """
    if not scores:
        return {}
    distinct = sorted({value for value in scores.values()}, reverse=True)
    span = len(distinct) - 1
    level = {value: 1.0 if step == 0 else 1.0 - step / span for step, value in enumerate(distinct)}
    return {item: level[value] for item, value in scores.items()}


def top_rank_normalised(scores: dict[str, float], limit: int) -> dict[str, float]:
    """Rank-normalise only a route's leaders, so position among them is what transfers.

    A phrase can match thousands of products. Rescaling all of them and then keeping
    the best few would leave the survivors inside a hairline of each other, which is
    another way of saying the phrase only tells whether a product matched at all.
    Keeping the leaders first and scoring them against each other is both cheaper and
    honest about what the route is claiming, and it is the same scale a dense route
    already returns.
    """
    if not scores:
        return {}
    limit = max(1, int(limit))
    leaders = heapq.nsmallest(limit, scores, key=lambda item: (-scores[item], item))
    count = len(leaders)
    return {item: (count - position) / count for position, item in enumerate(leaders)}


@dataclass(frozen=True)
class Prepared:
    """What a request's own history says, independent of anything that was said."""

    base: dict[str, float]
    base_order: tuple[str, ...]
    personalised: dict[str, float]
    eligible: list[str]
    seen: frozenset[str]
    allowed: frozenset[str] | None = None


@dataclass(frozen=True)
class TextRanker:
    """Committed route plus a frozen-weight adjustment from one utterance."""

    model: Recommender
    index: LexicalIndex | None = None
    alpha: float = 0.75
    baseline: str = "recent"
    weight: float = 0.5
    repulsion: float = 0.0
    price_penalty: float = 0.0
    enforce_ceiling: bool = False
    candidate_cap: int = 200
    embeddings: object | None = None
    scope: str = "catalogue"
    scope_depth: int = 2000

    def _personalised(self, history: tuple[str, ...], seen: set[str]) -> dict[str, float]:
        personalised: dict[str, float] = defaultdict(float)
        for source in last_distinct(history):
            for item, similarity in self.model.neighbors.get(source, ()):
                if item in self.model.recent and item not in seen:
                    personalised[item] += similarity
        return personalised

    def prepare(self, history: tuple[str, ...]) -> Prepared:
        history = tuple(history)
        seen = set(history)
        if self.baseline == "lifetime":
            base, base_order = self.model.lifetime, self.model.lifetime_order
        else:
            base, base_order = self.model.recent, self.model.recent_order
        eligible = [item for item in base_order if item not in seen]
        personalised = self._personalised(history, seen)
        return Prepared(base, base_order, personalised, eligible, frozenset(seen),
                        self._allowed(eligible, personalised))

    def _allowed(self, eligible: list[str], personalised: dict[str, float]) -> frozenset[str] | None:
        """Which products a phrase may promote at all.

        "catalogue" lets a phrase reach any product on the shelf. "behavioural"
        confines it to what the behavioural route was already considering: the
        co-review neighbours of this person's history, plus the products at the head
        of the fallback order. The second is how a large feed answers a spoken
        request — the words choose between plausible products rather than
        resurrecting something no one would have suggested.
        """
        if self.scope != "behavioural":
            return None
        allowed = set(eligible[:self.scope_depth])
        allowed.update(personalised)
        return frozenset(allowed)

    def phrase_scores(self, utterance: Utterance, eligible: set[str],
                      allowed: frozenset[str] | None = None):
        """Rank-normalised attraction, repulsion and price flags for one phrase.

        Every retrieval route is put on the same 0–1 scale before the two are
        fused, and the fusion takes the better of the two: an item that either
        route ranks highly counts as a match. What survives is capped to the
        widest pool the ranker is willing to re-order. `allowed`, when given, is
        the set of products a phrase may promote at all.
        """
        if utterance.empty or (self.index is None and self.embeddings is None):
            return {}, {}, set()
        routes = []
        if self.embeddings is not None:
            text = utterance.raw or " ".join(utterance.positive)
            retrieve = getattr(self.embeddings, "top_scores", None)
            if retrieve is not None:
                scores = dict(retrieve(text, self.candidate_cap, allowed)
                              if allowed is not None else retrieve(text, self.candidate_cap))
            else:
                scores = dict(self.embeddings.query_scores(text))
            routes.append({item: value for item, value in scores.items()
                           if value > 0.0 and (allowed is None or item in allowed)})
        if self.index is not None:
            scores = self.index.score(utterance.positive) or {}
            if allowed is not None:
                scores = {item: value for item, value in scores.items() if item in allowed}
            routes.append(scores)
        attraction: dict[str, float] = {}
        for scores in routes:
            for item, value in top_rank_normalised(scores, self.candidate_cap).items():
                attraction[item] = max(attraction.get(item, 0.0), value)
        if len(attraction) > self.candidate_cap:
            keep = heapq.nsmallest(self.candidate_cap, attraction,
                                   key=lambda item: (-attraction[item], item))
            attraction = {item: attraction[item] for item in keep}
        repulsion = {}
        if utterance.negative and self.index is not None:
            negative = self.index.score(utterance.negative)
            if negative:
                repulsion = top_rank_normalised(negative, self.candidate_cap)
        overpriced = set()
        if utterance.price_ceiling is not None and self.index is not None:
            overpriced = {item for item in eligible
                          if (price := self.index.price_of(item)) is not None
                          and price > utterance.price_ceiling}
        return attraction, repulsion, overpriced

    def route(self, utterance: Utterance, prepared: Prepared):
        """What a wording of a request pulls towards and pushes away.

        Run once per request and wording, then handed to every scoring
        configuration: what a phrase retrieves does not depend on how hard its
        results are trusted. Price is deliberately left out — it is checked against
        the pool in `rank_from`, because testing every catalogue price to reject
        half the catalogue costs more than testing the few hundred products a pool
        can hold, and both give the same order.
        """
        attraction, repulsion, _ = self.phrase_scores(utterance, frozenset(), prepared.allowed)
        return attraction, repulsion

    def over_ceiling(self, item: str, ceiling: float | None) -> bool:
        if ceiling is None or self.index is None or self.price_penalty == 0.0:
            return False
        price = self.index.price_of(item)
        return price is not None and price > ceiling

    def outside_budget(self, item: str, ceiling: float | None) -> bool:
        """Whether a product is known to cost more than a stated budget.

        A person who names a figure has stated a constraint, not a preference, so the
        serving route treats it as one: a product whose recorded price is above the
        figure is not offered at all. Products with no recorded price survive, because
        the archive is silent about them rather than contrary. The measured arm leaves
        this switch off and only scores the figure, which is what its frozen
        configurations were selected against.
        """
        if ceiling is None or self.index is None:
            return False
        price = self.index.price_of(item)
        return price is not None and price > ceiling

    def rank_from(self, prepared: Prepared, attraction: dict[str, float],
                  repulsion: dict[str, float], ceiling: float | None = None,
                  k: int = 10) -> tuple[str, ...]:
        """Order the candidates a phrase put in play, exactly as `rank` would.

        A product carrying no adjustment keeps its base order, so the top of the
        eligible list outranks anything deeper than it. The only way a deeper
        product gets in is by being lifted, so every lifted product joins the pool.
        The only way a listed product gets pushed out is by being depressed, so the
        list is extended by as many products as it depressed, until that stops
        changing. That keeps the order exact without touching the rest of the
        catalogue.
        """
        base, eligible = prepared.base, prepared.eligible
        personalised = prepared.personalised
        cached: dict[str, float] = {}

        def adjustment(item: str) -> float:
            if item not in cached:
                value = (self.weight * attraction.get(item, 0.0)
                         - self.repulsion * repulsion.get(item, 0.0))
                if self.over_ceiling(item, ceiling):
                    value -= self.price_penalty
                cached[item] = value
            return cached[item]

        take = min(len(eligible), k)
        while True:
            prefix = eligible[:take]
            depressed = sum(1 for item in prefix if adjustment(item) < 0.0)
            refused = (sum(1 for item in prefix if self.outside_budget(item, ceiling))
                       if self.enforce_ceiling else 0)
            if take >= k + depressed + refused or take >= len(eligible):
                break
            take = min(len(eligible), k + depressed + refused)
        pool = set(prefix)
        pool.update(item for item, value in attraction.items() if value > 0.0)
        pool.update(personalised)
        # A phrase can name something the person already owns, so the history is
        # removed after the phrase has had its say, not before.
        pool.difference_update(prepared.seen)
        if self.enforce_ceiling and ceiling is not None:
            # Honouring a stated figure beats answering at all costs, but silence is
            # better than nothing: if every candidate is known to be too expensive,
            # the figure is dropped rather than returning an empty answer.
            affordable = {item for item in pool if not self.outside_budget(item, ceiling)}
            if affordable:
                pool = affordable

        def score(item: str) -> float:
            return ((1 - self.alpha) * base[item] + self.alpha * personalised.get(item, 0.0)
                    + adjustment(item))

        return tuple(sorted(pool, key=lambda item: (-score(item), -base[item], item))[:k])

    def rank(self, history, phrase: str | Utterance = "", k: int = 10) -> tuple[str, ...]:
        history = tuple(history)
        utterance = phrase if isinstance(phrase, Utterance) else parse_utterance(str(phrase or ""))
        if (self.weight == 0.0 or utterance.empty) and self.price_penalty == 0.0:
            return self.model.top_k(history, self.baseline, self.alpha, k)
        prepared = self.prepare(history)
        attraction, repulsion = self.route(utterance, prepared)
        return self.rank_from(prepared, attraction, repulsion, utterance.price_ceiling, k)
