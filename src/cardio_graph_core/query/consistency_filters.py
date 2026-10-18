"""Deterministic consistency checks between a patient mention and a guideline condition.

Used by
  * verified_logic_evaluator.py  (live: --logic-module cardio_graph_core.query.verified_logic_evaluator)
  * analysis/kg_posthoc_filters.py (offline: re-evaluates an --include-debug trace)

Both give identical results. No LLM, no graph access, no disease-specific rules.

A patient evidence item may support a decision node only if it passes the enabled checks:

  certainty  ConText-style certainty agreement [Harkema 2009]. An uncertainty cue
             ("suspected", "possible", "suggestive of", "X is suspected", ...) that scopes over
             the words the mention and the condition share must be present on both sides or
             on neither: "suspected X" supports "individuals with suspected X" but not
             "patients with established X", and an asserted "X" does not support the
             suspected-X work-up rules. "documented or suspected X" accepts both.
  negation   NegEx/ConText negation scope in the vignette [Chapman 2001]: a mention under a
             negation cue in its clause never supports a PRESENT/numeric condition, even if
             the extractor dropped the negation word.
  qualifier  Entailment direction on generic qualifier axes (chronic/acute/subacute,
             stable/unstable): a condition that carries a qualifier is supported only by a
             mention that carries it too ("myocardial ischaemia" does not entail "chronic
             myocardial ischaemia"); a more specific mention still entails a general condition.
  absence    Explicit absence (the reference standard's semantics): ABSENT is true only if
             the vignette negates a mention grounded to that concept (default off: the
             original run uses closed-world absence).

Scope: a cue scopes forward to the next terminator (comma, semicolon, "and", "with", "but",
...) or, for post-cues, over the five preceding words; it never crosses into a neighbouring
finding. The cue lists are generic (ConText/NegEx families) and contain no disease names.
They were written after the error analysis of the KG-100 run, so report results with these
filters as a post-hoc analysis.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

DEFAULT_FILTERS = ("certainty", "negation", "qualifier")
ALL_FILTERS = ("certainty", "negation", "qualifier", "absence")

UNCERTAIN_PRE = [
    r"suspected", r"suspect", r"suspicion of", r"possible", r"possibly", r"probable",
    r"probably", r"presumed", r"presumptive", r"likely", r"questionable", r"query",
    r"rule out", r"r/o", r"suggestive of", r"concerning for", r"cannot be excluded",
    r"under evaluation for", r"evaluation for", r"work-?up for",
]
UNCERTAIN_POST = [r"is suspected", r"are suspected", r"was suspected", r"cannot be excluded",
                  r"is possible", r"is likely", r"is questionable"]
CERTAIN_CUES = [r"established", r"documented", r"confirmed", r"known", r"proven",
                r"diagnosed"]
NEGATION_PRE = [r"no", r"not", r"without", r"absence of", r"absent", r"denies", r"denied",
                r"negative for", r"free of", r"never"]
CUE_WORDS = {"suspected", "suspect", "suspicion", "possible", "possibly", "probable",
             "probably", "presumed", "presumptive", "likely", "questionable", "query"}
TERMINATOR = re.compile(
    r"[,;:.()]|\b(?:and|but|however|although|which|while|whereas|with)\b", re.I)
QUALIFIER_AXES = {
    "chronicity": ("chronic", "acute", "subacute"),
    "stability": ("stable", "unstable"),
}
STOP = {"of", "the", "a", "an", "in", "with", "and", "or", "to", "for", "on", "at", "by",
        "patients", "patient", "individuals", "individual", "who", "is", "are", "be"}


def _cue_re(cues: list[str]) -> re.Pattern:
    return re.compile(r"(?<![\w-])(?:" + "|".join(cues) + r")(?![\w-])", re.I)


UNC_PRE_RE, UNC_POST_RE, NEG_RE = _cue_re(UNCERTAIN_PRE), _cue_re(UNCERTAIN_POST), _cue_re(NEGATION_PRE)
BOTH_RE = re.compile(
    r"(?:%s)\s+or\s+(?:suspected|possible|probable)|(?:suspected|possible|probable)\s+or\s+(?:%s)"
    % ("|".join(CERTAIN_CUES), "|".join(CERTAIN_CUES)), re.I)


# ---------------------------------------------------------------- text helpers
def words(s: Any) -> list[str]:
    s = re.sub(r"\s+", " ", str(s or "").lower())
    return re.findall(r"[a-z0-9]+", s.replace("ischaem", "ischem").replace("anaem", "anem"))


def content(s: Any) -> set[str]:
    return {w for w in words(s) if len(w) > 2 and w not in STOP}


def initialisms(text: str) -> dict[str, list[str]]:
    """'chronic coronary syndrome' -> {'ccs': [...]} for 2-4-word windows."""
    ws = words(text)
    out: dict[str, list[str]] = {}
    for n in (2, 3, 4):
        for i in range(len(ws) - n + 1):
            out.setdefault("".join(w[0] for w in ws[i:i + n]), ws[i:i + n])
    return out


def clause_of(text: str, start: int, end: int) -> tuple[str, str, str]:
    """(left context, mention, right context), each cut at the nearest terminator."""
    left = text[:start]
    cuts = [m.end() for m in TERMINATOR.finditer(left)]
    left = left[cuts[-1]:] if cuts else left
    right = text[end:]
    m = TERMINATOR.search(right)
    return left, text[start:end], (right[:m.start()] if m else right)


def locate(mention: str, text: str) -> list[tuple[int, int]]:
    """Spans of the mention in the vignette: exact match, else the shortest window that
    holds all content words of the mention (small paraphrases by the extractor)."""
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(mention.lower()), text.lower())]
    if spans:
        return spans
    want = content(mention)
    if not want:
        return []
    toks = [(m.group(0).lower().replace("ischaem", "ischem"), m.start(), m.end())
            for m in re.finditer(r"[A-Za-z0-9]+", text)]
    best = None
    for i in range(len(toks)):
        seen: set[str] = set()
        for j in range(i, min(len(toks), i + 25)):
            if toks[j][0] in want:
                seen.add(toks[j][0])
            if seen == want:
                if best is None or toks[j][2] - toks[i][1] < best[1] - best[0]:
                    best = (toks[i][1], toks[j][2])
                break
    return [best] if best else []


# ---------------------------------------------------------------- checks
def mention_status(mention: str, vignette: str | None) -> dict[str, Any]:
    """Clause(s) of the mention in the vignette and whether it is negated there.
    Without a vignette (oracle mentions) the mention text itself is the clause."""
    spans = locate(mention, vignette) if vignette else []
    if not spans:
        neg = bool(re.match(r"\s*(?:no|not|without)\b", mention, re.I))
        return {"found": False, "negated": neg, "clauses": [mention]}
    neg, clauses = None, []
    for s, e in spans:  # a mention found twice counts as negated only if all are
        left, mid, right = clause_of(vignette, s, e)
        n = bool(NEG_RE.search(left) or re.match(r"\s*(?:no|not|without)\b", mid, re.I))
        neg = n if neg is None else (neg and n)
        clauses.append(left + mid + right)
    return {"found": True, "negated": neg, "clauses": clauses}


def uncertain_over(text: str, targets: set[str], inits: set[str] = frozenset()) -> bool:
    """Does an uncertainty cue in `text` scope over one of the target words?"""
    for m in UNC_PRE_RE.finditer(text):
        scope = text[m.end():]
        t = TERMINATOR.search(scope)
        sw = words(scope[:t.start()] if t else scope)
        if targets & set(sw) or any(w in inits for w in sw):
            return True
    for m in UNC_POST_RE.finditer(text):
        if targets & set(words(" ".join(text[:m.start()].split()[-5:]))):
            return True
    return False


def certainty_ok(mention: str, clauses: list[str], concept: str, context: str) -> bool:
    concept, context = concept or "", context or ""
    if BOTH_RE.search(context) or BOTH_RE.search(concept):
        return True
    cw, mw = content(concept) - CUE_WORDS, content(mention) - CUE_WORDS
    shared = cw & mw
    node_t = shared or set([w for w in words(concept) if w in cw][:1])
    ment_t = shared or set([w for w in words(mention) if w in mw][:1])
    node_unc = uncertain_over(concept, node_t) or uncertain_over(context, node_t, set(initialisms(concept)))
    ment_unc = all(uncertain_over(c, ment_t) for c in clauses) if clauses else False
    return node_unc == ment_unc


def _qualifiers(text: str) -> dict[str, set[str]]:
    ws = set(words(text))
    return {ax: {q for q in qs if q in ws} for ax, qs in QUALIFIER_AXES.items()}


def qualifier_ok(concept: str, context: str, clauses: list[str]) -> bool:
    """Every qualifier carried by the condition's concept must be carried by the mention
    clause; an abbreviation in the clause ('CCS') inherits the qualifiers of the expansion
    it abbreviates in the condition text."""
    need = _qualifiers(concept or "")
    if not any(need.values()):
        return True
    node_inits = initialisms(f"{concept} {context or ''}")
    for clause in clauses:
        have = _qualifiers(clause)
        for tok in re.findall(r"\b[A-Z]{2,6}\b", clause):
            exp = node_inits.get(tok.lower())
            if exp:
                for ax, qs in _qualifiers(" ".join(exp)).items():
                    have[ax] |= qs
        if all(not qs or (qs & have[ax]) for ax, qs in need.items()):
            return True
    return False


def concept_name(decision: dict[str, Any]) -> str:
    """Same concept name the base evaluator uses to look up evidence."""
    c = decision.get("concept")
    if isinstance(c, dict):
        return (c.get("entity_standardized_candidate") or decision.get("entity_standardized_candidate")
                or c.get("entity") or decision.get("entity") or "")
    return c or decision.get("entity_standardized_candidate") or decision.get("entity") or ""


class Checker:
    """Per-case checker. `vignette` may be None (oracle mentions)."""

    def __init__(self, vignette: str | None, filters=DEFAULT_FILTERS) -> None:
        self.vignette = vignette
        self.filters = set(filters)
        unknown = self.filters - set(ALL_FILTERS)
        if unknown:
            raise ValueError(f"unknown filters: {sorted(unknown)}")
        self._st: dict[str, dict] = {}

    def status(self, mention: str) -> dict[str, Any]:
        if mention not in self._st:
            self._st[mention] = mention_status(mention, self.vignette)
        return self._st[mention]

    def rejection(self, evidence: dict[str, Any], concept: str, context: str) -> str | None:
        """None if the evidence may support the condition, else the failing check."""
        mention = evidence.get("query_entity") or evidence.get("original_text") or ""
        st = self.status(mention)
        if "negation" in self.filters and st["negated"]:
            return "negation"
        if "certainty" in self.filters and not certainty_ok(mention, st["clauses"], concept, context):
            return "certainty"
        if "qualifier" in self.filters and not qualifier_ok(concept, context, st["clauses"]):
            return "qualifier"
        return None


def combine_groups(evaluated: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """Unchanged rule logic of logic_evaluator.evaluate_rule: AND/OR/null per logic group,
    all groups must hold."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in evaluated:
        groups[str(d.get("logic_group") or "g0")].append(d)
    results = []
    for gid, nodes in sorted(groups.items()):
        lt = lambda n: (n.get("logic_type") or "").upper()  # noqa: E731
        a = [n for n in nodes if lt(n) == "AND"]
        o = [n for n in nodes if lt(n) == "OR"]
        x = [n for n in nodes if lt(n) not in {"AND", "OR"}]
        ok = (all(n["truth"] for n in a) if a else True) and (any(n["truth"] for n in o) if o else True) \
            and (all(n["truth"] for n in x) if x else True)
        results.append({"logic_group": gid, "group_true": ok, "and_nodes": a, "or_nodes": o, "null_nodes": x})
    return (all(g["group_true"] for g in results) if results else False), results


def evaluate_decisions(decisions: list[dict[str, Any]], evidence: list[dict[str, Any]],
                       checker: Checker, base: Any, negated_concepts: set[str] | None = None):
    """Evaluate each decision with the evidence that passes the checks for that decision,
    using the base module's evaluate_decision_node. Returns (evaluated, rule_true,
    group_results, reasons) where reasons lists the checks that removed supporting evidence."""
    evaluated, reasons = [], []
    for d in decisions:
        concept, context = concept_name(d), d.get("context") or ""
        op = base.normalize_operator(d.get("operator"))
        if op == "ABSENT" and "absence" in checker.filters:
            truth = base._norm_text(concept) in (negated_concepts or set())
            res = base.evaluate_decision_node(d, {})  # fills the standard fields
            res.update(truth=truth, matched_evidence=[], reason=(
                "concept explicitly negated in the case" if truth else "absence not stated (explicit absence)"))
            if not truth and base.evaluate_decision_node(d, base.build_evidence_index(evidence))["truth"]:
                reasons.append("absence")
            evaluated.append(res)
            continue
        kept, dropped = [], []
        for e in evidence:
            why = checker.rejection(e, concept, context)
            (kept if why is None else dropped).append((e, why))
        res = base.evaluate_decision_node(d, base.build_evidence_index([e for e, _ in kept]))
        if not res["truth"]:
            full = base.evaluate_decision_node(d, base.build_evidence_index(evidence))
            if full["truth"]:
                key = base._norm_text(concept)
                reasons += sorted({why for e, why in dropped if base._norm_text(e.get("standardized_concept")) == key})
                res["reason"] = f"{res['reason']} (evidence rejected by consistency check: {', '.join(sorted(set(reasons))) or '-'})"
        evaluated.append(res)
    rule_true, group_results = combine_groups(evaluated)
    return evaluated, rule_true, group_results, sorted(set(reasons))


def negated_concepts_from_matches(matched_nodes: list[dict[str, Any]], checker: Checker, base: Any) -> set[str]:
    out = set()
    for mn in matched_nodes or []:
        q = mn.get("query_entity") or ""
        if base.is_negated_text(q) or checker.status(q)["negated"]:
            out.add(base._norm_text(mn.get("matched_node")))
    return out
