"""
Stage 1a real-document ingestion: extract claims from real text with an LLM,
so the contradiction engine can run on extracted (not hand-authored) claims.

Design constraints (interpretability first):
  - The LLM extracts CLAIMS and proposes RELATIONSHIPS.
  - Reliability is PROVIDED as document metadata, never LLM-judged.
  - Proposed relationships are written to a REVIEW FILE
    (data/proposed_relationships.json) for human approval; they are NOT fed
    to the engine automatically.

Request budget: a full run is 2 LLM requests — one batched claim-extraction
call over all documents, plus one relationship-proposal call — to stay within
tight free-tier quotas. If the batch response drops a document, that document is
re-extracted on its own so nothing is silently lost.

Output claim schema matches the hand-authored corpus exactly, so extracted
claims drop straight into graph_reasoning.ingest.load_into_graph:
    {claim_id/id, claim_text, topic, source, source_reliability, ground_truth}

Model: gemini-2.5-flash-lite via Google's Gemini API (override via
extract_claims(..., model=...)). Requires the `google-generativeai` package and
a key in GEMINI_API_KEY (or GOOGLE_API_KEY). If neither is available the call
fails clearly. Structured JSON output is requested via Gemini's JSON mode
(response_mime_type + response_schema) so parsing stays clean.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from .graph import ReasoningGraph, RelationType

DEFAULT_MODEL = "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# 1. Document input format
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A real document: raw text plus provided (not LLM-judged) metadata."""
    doc_id: str                 # stable id, derived from the filename stem
    text: str
    topic: str
    source: str
    source_reliability: float
    ground_truth: bool = False

    @classmethod
    def from_metadata_entry(cls, entry, base_dir):
        path = Path(base_dir) / entry["file"]
        return cls(
            doc_id=Path(entry["file"]).stem,
            text=path.read_text(),
            topic=entry["topic"],
            source=entry["source"],
            source_reliability=float(entry["source_reliability"]),
            ground_truth=bool(entry.get("ground_truth", False)),
        )


def load_documents(metadata_path):
    """Load a document set from a metadata sidecar + its text files."""
    metadata_path = Path(metadata_path)
    meta = json.loads(metadata_path.read_text())
    base_dir = metadata_path.parent
    return [Document.from_metadata_entry(e, base_dir) for e in meta["documents"]]


# ---------------------------------------------------------------------------
# Gemini client helper — fail clearly if unavailable
# ---------------------------------------------------------------------------

def _get_api_key():
    """Return the Gemini API key from GEMINI_API_KEY or GOOGLE_API_KEY, else None."""
    import os

    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _get_client():
    """
    Configure the Gemini SDK and return the `genai` module, or raise a clear
    error. The key is read from GEMINI_API_KEY (or GOOGLE_API_KEY) — we do NOT
    silently proceed keyless.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "The `google-generativeai` package is required for LLM extraction. "
            "Install it with `pip install google-generativeai`."
        ) from e

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "No Gemini credentials found. LLM extraction needs an API key. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY). Failing clearly rather "
            "than proceeding without credentials."
        )

    try:
        genai.configure(api_key=api_key)
    except Exception as e:  # noqa: BLE001 - surface the real cause to the caller
        raise RuntimeError(
            f"Could not configure the Gemini client (underlying error: {e})."
        ) from e
    return genai


# ---------------------------------------------------------------------------
# 2. LLM claim extraction (reliability comes from metadata, not the LLM)
# ---------------------------------------------------------------------------

class ExtractedClaim(BaseModel):
    """One atomic claim the LLM found in a document."""
    claim_text: str = Field(
        description="A short, atomic statement of one claim the document asserts."
    )


class ExtractionResult(BaseModel):
    """Structured extraction output for a single document."""
    claims: List[ExtractedClaim] = Field(
        description="The atomic claims asserted by the document."
    )


class DocumentClaims(BaseModel):
    """Claims extracted for one document, tagged with its doc_id."""
    doc_id: str = Field(description="The document id these claims came from.")
    claims: List[ExtractedClaim] = Field(
        description="The atomic claims asserted by this document."
    )


class BatchExtractionResult(BaseModel):
    """Structured extraction output for a batch of documents."""
    documents: List[DocumentClaims] = Field(
        description="Per-document claim lists, one entry per input document."
    )


_EXTRACTION_SYSTEM = (
    "You extract atomic factual claims from a document. A claim is a single, "
    "self-contained assertion the document makes about the world. Split compound "
    "statements into separate atomic claims. Extract only what the document "
    "actually asserts — do not add outside knowledge, do not judge whether a "
    "claim is true, and do not assess the reliability of the source. Keep each "
    "claim short (one sentence)."
)

_BATCH_EXTRACTION_SYSTEM = _EXTRACTION_SYSTEM + (
    " You are given several documents, each with a doc_id. Return one entry per "
    "document, tagging each with the exact doc_id it was given, and list that "
    "document's atomic claims under it. Do not merge claims across documents."
)


def _structured_call(client, model, system, message, schema_model, tool_name=None,
                     max_output_tokens=3000):
    """
    Get schema-validated JSON from Gemini. Uses JSON mode (response_mime_type +
    response_schema) so the model returns JSON matching `schema_model`, which we
    then validate with Pydantic. `client` is the configured `genai` module;
    `system` is passed as the model's system instruction. `tool_name` is unused
    (kept for call-site compatibility). Returns a schema_model instance.
    """
    import time

    from google.api_core import exceptions as g_exc

    generative_model = client.GenerativeModel(
        model_name=model,
        system_instruction=system,
    )
    config = {
        "response_mime_type": "application/json",
        "response_schema": schema_model,
        "max_output_tokens": max_output_tokens,
    }

    # Retry on 429 (free-tier per-minute rate limit), honoring the API's own
    # retry_delay when present. A few paced retries clear the 5-rpm free tier;
    # a persistent quota-exhausted error is reported clearly after the attempts.
    max_retries = 6
    for attempt in range(max_retries + 1):
        try:
            response = generative_model.generate_content(message, generation_config=config)
            return _lenient_validate(schema_model, response.text)
        except g_exc.ResourceExhausted as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Gemini quota exhausted for model {model!r} after "
                    f"{max_retries} retries. This is usually the per-day "
                    "free-tier limit — retry after the daily reset (midnight "
                    "Pacific), enable billing on the API key's project, or use "
                    f"a model with remaining quota. (Underlying error: {e})"
                ) from e
            suggested = getattr(getattr(e, "retry_delay", None), "seconds", 0) or 0
            time.sleep(max(suggested, 20) + 1)
        except g_exc.GoogleAPICallError as e:
            # Not-found model, bad schema, permission, etc. — fail clearly.
            raise RuntimeError(
                f"Gemini API call failed for model {model!r}: {e}"
            ) from e


# String fields that models sometimes omit and which we tolerate by backfilling
# "" (they carry explanation, not identity). Backfilling is restricted to these
# by name so genuinely required fields (from_id/to_id/type/claim_text) still
# fail validation when absent — an empty id must never slip through.
_OPTIONAL_STRING_FIELDS = {"rationale"}


def _lenient_validate(schema_model, text):
    """
    Validate Gemini's JSON against schema_model, tolerating omitted explanatory
    string fields listed in _OPTIONAL_STRING_FIELDS (e.g. a missing `rationale`).
    Load-bearing fields still fail validation if absent. Backfilling applies to
    the top object and to any list-of-objects it contains (e.g. relationships).
    """
    data = json.loads(text)

    def backfill(model_cls, obj):
        if not isinstance(obj, dict):
            return
        for fname, field in model_cls.model_fields.items():
            if field.annotation is str and fname in _OPTIONAL_STRING_FIELDS:
                obj.setdefault(fname, "")
            value = obj.get(fname)
            item_model = getattr(field.annotation, "__args__", (None,))[0]
            if isinstance(value, list) and hasattr(item_model, "model_fields"):
                for item in value:
                    backfill(item_model, item)

    backfill(schema_model, data)
    return schema_model.model_validate(data)


def extract_claims_from_document(doc, client=None, model=DEFAULT_MODEL):
    """
    Extract atomic claims from one document via the LLM, then attach the
    PROVIDED metadata (topic, source, source_reliability, ground_truth).
    Returns a list of claim dicts in the hand-authored corpus schema.
    """
    client = client or _get_client()

    message = (
        f"Document topic: {doc.topic}\n\n"
        f"Document text:\n{doc.text.strip()}\n\n"
        "Extract the atomic claims this document asserts."
    )

    result = _structured_call(
        client, model, _EXTRACTION_SYSTEM, message,
        ExtractionResult, "record_claims",
    )

    claims = []
    for i, extracted in enumerate(result.claims):
        claims.append({
            "id": f"{doc.doc_id}_c{i}",
            "claim_text": extracted.claim_text,
            "topic": doc.topic,
            "source": doc.source,               # provided metadata
            "source_reliability": doc.source_reliability,  # provided, not LLM-judged
            "ground_truth": doc.ground_truth,   # provided metadata
            "doc_id": doc.doc_id,
        })
    return claims


def _claim_dict(doc, index, claim_text):
    """Assemble one claim dict in the hand-authored corpus schema for `doc`."""
    return {
        "id": f"{doc.doc_id}_c{index}",
        "claim_text": claim_text,
        "topic": doc.topic,
        "source": doc.source,               # provided metadata
        "source_reliability": doc.source_reliability,  # provided, not LLM-judged
        "ground_truth": doc.ground_truth,   # provided metadata
        "doc_id": doc.doc_id,
    }


def extract_claims(documents, client=None, model=DEFAULT_MODEL, verbose=False):
    """
    Extract claims across a set of documents in a SINGLE batched LLM call, then
    attach each document's PROVIDED metadata by matching doc_id. Returns a flat
    list of claim dicts in the hand-authored corpus schema. One request for the
    whole set keeps the run within tight free-tier quotas (vs. one per document).
    """
    client = client or _get_client()
    documents = list(documents)
    by_id = {doc.doc_id: doc for doc in documents}

    catalog = "\n\n".join(
        f"doc_id: {doc.doc_id}\ntopic: {doc.topic}\ntext:\n{doc.text.strip()}"
        for doc in documents
    )
    message = (
        "Extract atomic claims for each of the following documents. Return one "
        "entry per document, tagged with its exact doc_id.\n\n" + catalog
    )

    result = _structured_call(
        client, model, _BATCH_EXTRACTION_SYSTEM, message,
        BatchExtractionResult, "record_batch_claims",
        max_output_tokens=8000,
    )

    all_claims = []
    seen = set()
    for entry in result.documents:
        doc = by_id.get(entry.doc_id)
        if doc is None:
            continue  # ignore any doc_id the model invented
        seen.add(entry.doc_id)
        claims = [_claim_dict(doc, i, c.claim_text)
                  for i, c in enumerate(entry.claims)]
        if verbose:
            print(f"  {doc.doc_id}: extracted {len(claims)} claim(s)")
        all_claims.extend(claims)

    # Fallback: if the batch response dropped a document, extract it on its own
    # so no document is silently lost.
    for doc in documents:
        if doc.doc_id not in seen:
            if verbose:
                print(f"  {doc.doc_id}: missing from batch, extracting individually")
            all_claims.extend(
                extract_claims_from_document(doc, client=client, model=model))

    return all_claims


def _extract_claims_per_document(documents, client=None, model=DEFAULT_MODEL,
                                 verbose=False):
    """Per-document extraction (one request each). Kept for reference/fallback."""
    client = client or _get_client()
    all_claims = []
    for doc in documents:
        claims = extract_claims_from_document(doc, client=client, model=model)
        if verbose:
            print(f"  {doc.doc_id}: extracted {len(claims)} claim(s)")
        all_claims.extend(claims)
    return all_claims


# ---------------------------------------------------------------------------
# 3. LLM relationship proposal (written to a review file, NOT fed to engine)
# ---------------------------------------------------------------------------

class ProposedRelationship(BaseModel):
    """A support/contradiction relationship the LLM proposes between two claims."""
    from_id: str = Field(description="The id of the source claim.")
    to_id: str = Field(description="The id of the target claim.")
    type: str = Field(description='Either "support" or "contradiction".')
    # Kept as a plain required field so the schema Gemini receives contains NO
    # `default` key (Gemini's response_schema rejects `default`). The model
    # sometimes omits it anyway; _structured_call fills a "" before validation
    # so an otherwise-valid relationship is not dropped over a missing rationale.
    rationale: str = Field(description="One short sentence explaining the link.")


class RelationshipProposal(BaseModel):
    relationships: List[ProposedRelationship] = Field(
        description="Proposed support/contradiction relationships between the claims."
    )


_RELATIONSHIP_SYSTEM = (
    "You are given a list of claims, each with an id, its topic, and its text. "
    "Propose support and contradiction relationships BETWEEN claims. Use "
    '"contradiction" when two claims cannot both be true, and "support" when '
    "one claim provides evidence for or reinforces another. Only relate claims "
    "within the same topic. Do not relate a claim to itself. Only propose "
    "relationships you are confident about based solely on the claim texts. "
    "These proposals will be reviewed by a human before use."
)


def propose_relationships(claims, client=None, model=DEFAULT_MODEL):
    """
    Ask the LLM to propose support/contradiction relationships among the claims.
    Returns a list of proposal dicts: {from, to, type, rationale}.
    """
    client = client or _get_client()

    catalog = "\n".join(
        f"- id={c['id']} | topic={c['topic']} | {c['claim_text']}"
        for c in claims
    )
    message = (
        "Claims:\n" + catalog + "\n\n"
        "Propose support/contradiction relationships between these claims."
    )

    # Relationship JSON over many claims can be long — give it ample room so the
    # response isn't truncated mid-object (which would fail JSON parsing).
    proposal = _structured_call(
        client, model, _RELATIONSHIP_SYSTEM, message,
        RelationshipProposal, "record_relationships",
        max_output_tokens=8000,
    )

    valid_ids = {c["id"] for c in claims}
    out = []
    for rel in proposal.relationships:
        if rel.from_id not in valid_ids or rel.to_id not in valid_ids:
            continue  # drop hallucinated ids
        if rel.type not in ("support", "contradiction"):
            continue
        if rel.from_id == rel.to_id:
            continue
        out.append({
            "from": rel.from_id,
            "to": rel.to_id,
            "type": rel.type,
            "rationale": rel.rationale,
        })
    return out


def write_proposed_relationships(claims, relationships, path):
    """
    Write extracted claims + proposed relationships to a review file for human
    approval. This is the human-in-the-loop gate: nothing here reaches the
    engine until a person inspects and approves it.
    """
    path = Path(path)
    payload = {
        "description": (
            "REVIEW REQUIRED. LLM-extracted claims and LLM-PROPOSED relationships. "
            "Inspect and approve the relationships before running the engine. "
            "Reliability is from document metadata, not LLM-judged."
        ),
        "status": "proposed_unverified",
        "claims": claims,
        "relationships": relationships,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# 4. Loader: extracted claims + VERIFIED relationships -> ReasoningGraph
# ---------------------------------------------------------------------------

_RELATION_MAP = {
    "support": RelationType.SUPPORT,
    "contradiction": RelationType.CONTRADICTION,
}


def load_extracted_into_graph(claims, relationships, graph=None,
                              name="extracted-corpus"):
    """
    Build a ReasoningGraph from extracted claims and (human-verified)
    relationships. Mirrors graph_reasoning.ingest.load_into_graph: node
    confidence is seeded from source_reliability; source/topic/ground_truth go
    into node metadata; each relationship becomes a typed edge weighted by the
    source claim's reliability.
    """
    if graph is None:
        graph = ReasoningGraph(name)

    for claim in claims:
        graph.add_thought(
            claim["id"],
            claim["claim_text"],
            confidence=claim["source_reliability"],
            metadata={
                "source": claim["source"],
                "source_reliability": claim["source_reliability"],
                "topic": claim["topic"],
                "ground_truth": claim.get("ground_truth", False),
                "doc_id": claim.get("doc_id"),
            },
        )

    for rel in relationships:
        rel_type = _RELATION_MAP.get(rel["type"])
        if rel_type is None:
            raise ValueError(f"Unknown relationship type: {rel['type']!r}")
        from_id, to_id = rel["from"], rel["to"]
        edge_conf = graph.graph.nodes[from_id]["metadata"].get(
            "source_reliability", 0.5)
        graph.add_relation(from_id, to_id, rel_type, confidence=edge_conf)

    return graph
