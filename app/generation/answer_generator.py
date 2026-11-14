import json
import os
import mimetypes
import re
import time
import argparse

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

from app.retrieval.vector_retriever import retrieve_by_query
from app.retrieval.context_builder import build_multimodal_context


# CONFIGURATION

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)

MAX_IMAGES = 4

# =====================================================
# VERIFICATION CHECKPOINT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PENDING_VERIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "verification"
    / "pending_verification.json"
)

# =====================================================
# GEMINI API RETRY HANDLER
# =====================================================

def generate_content_with_retries(
    contents,
    config,
    max_retries=3
):
    """
    Generate Gemini content with retries for temporary
    API errors.

    Does not repeatedly retry daily quota exhaustion.
    """

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"\nGemini API attempt: "
                f"{attempt}/{max_retries}"
            )

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=contents,

                config=config

            )

            return response

        except errors.APIError as error:

            status_code = getattr(error, "code", None)
            error_message = str(error)

            print(f"\nGemini API error: {error_message}")
            print(f"HTTP status code: {status_code}")

            # Stop immediately for daily quota exhaustion.

            if status_code == 429 and any(
                keyword in error_message.lower()
                for keyword in [
                    "perday",
                    "per day",
                    "daily",
                    "generate_requests_per_day"
                ]
            ):

                print(
                    "Daily Gemini quota exhausted. "
                    "Stopping retries."
                )

                raise

            # Retry temporary server errors.

            retryable_codes = [
                429,
                500,
                502,
                503,
                504
            ]

            if (
                status_code not in retryable_codes
                or attempt == max_retries
            ):

                raise

            wait_time = 2 ** attempt

            print(
                f"Temporary API error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Gemini generation failed after all retry attempts."
    )

# =====================================================
# IMAGE LOADING
# =====================================================

def load_image_part(image_path: str):

    path = Path(image_path)

    if not path.exists():
        print(f"WARNING: Image not found: {path}")
        return None

    mime_type, _ = mimetypes.guess_type(str(path))

    if not mime_type or not mime_type.startswith("image/"):
        print(f"WARNING: Unsupported image: {path}")
        return None

    try:

        with open(path, "rb") as image_file:
            image_bytes = image_file.read()

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

    except Exception as error:

        print(f"Image loading failed: {error}")
        return None


# EVIDENCE REGISTRY

def build_evidence_registry(retrieved_chunks: list) -> dict:
    """
    Build a registry containing the exact PDF evidence
    retrieved from PostgreSQL.

    Every evidence item must have a valid chunk ID.

    Additionally, classify evidence based on its precision:

    - table             -> exact_structured
    - text              -> textual
    - embedded_image    -> approximate_visual
    """

    evidence_registry = {}

    for chunk in retrieved_chunks:

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            print(
                "WARNING: Retrieved chunk without chunk_id. "
                "Skipping untraceable evidence."
            )
            continue

        if chunk_id in evidence_registry:
            continue

        # ---------------------------------
        # Extract content type
        # ---------------------------------

        content_type = str(
            chunk.get("content_type", "unknown")
        ).lower()

        # ---------------------------------
        # Classify evidence precision
        # ---------------------------------

        if content_type == "table":

            precision = "exact_structured"
            evidence_priority = 1

        elif content_type == "text":

            precision = "textual"
            evidence_priority = 2

        elif content_type in [
            "visual",
            "embedded_image",
            "image"
        ]:

            precision = "approximate_visual"
            evidence_priority = 3

        else:

            precision = "unknown"
            evidence_priority = 4

        # ---------------------------------
        # Build evidence registry
        # ---------------------------------

        evidence_registry[chunk_id] = {

            "chunk_id": chunk_id,

            "document_name": chunk.get(
                "document_name",
                "Unknown document"
            ),

            "page_number": chunk.get(
                "page_number",
                "Unknown"
            ),

            "content_type": content_type,

            "content": chunk.get(
                "content",
                ""
            ),

            "image_path": chunk.get(
                "image_path"
            ),

            "metadata": chunk.get(
                "metadata",
                {}
            ),

            "similarity": chunk.get(
                "similarity"
            ),

            # New fields
            "precision": precision,

            "evidence_priority": evidence_priority

        }

    print(
        f"\nEvidence registry created: "
        f"{len(evidence_registry)} evidence items"
    )

    return evidence_registry


# =====================================================
# DEBUG: INSPECT RETRIEVED EVIDENCE
# =====================================================

def inspect_evidence_registry(evidence_registry: dict):
    """
    Print actual retrieved evidence and metadata.

    This helps us understand the extraction structure
    before implementing detailed source references.
    """

    print("\n========== EVIDENCE AUDIT ==========")

    for evidence_id, evidence in evidence_registry.items():

        print("\n" + "=" * 60)

        print(f"EVIDENCE ID: {evidence_id}")

        print(f"DOCUMENT: {evidence.get('document_name')}")

        print(f"PAGE: {evidence.get('page_number')}")

        print(f"CONTENT TYPE: {evidence.get('content_type')}")

        print(f"PRECISION: {evidence.get('precision')}")

        print(
            f"EVIDENCE PRIORITY: "
            f"{evidence.get('evidence_priority')}"
        )

        print("\n--- METADATA ---")

        print(
            json.dumps(
                evidence.get("metadata", {}),
                indent=4,
                ensure_ascii=False,
                default=str
            )
        )

        print("\n--- CONTENT ---")

        content = evidence.get("content", "")

        print(
            str(content)[:5000]
        )

        print("\n--- IMAGE PATH ---")

        print(evidence.get("image_path"))

    print("\n========== END EVIDENCE AUDIT ==========\n")

# FORMAT EVIDENCE

def format_evidence(evidence_registry: dict) -> str:

    formatted_evidence = []

    for chunk_id, evidence in evidence_registry.items():

        block = f"""
----------------------------------------
EVIDENCE ID: {chunk_id}

DOCUMENT: {evidence['document_name']}

PAGE NUMBER: {evidence['page_number']}

CONTENT TYPE: {evidence['content_type']}

PRECISION: {evidence['precision']}

EVIDENCE PRIORITY: {evidence['evidence_priority']}

SIMILARITY: {evidence['similarity']}

CONTENT:
{evidence['content']}

----------------------------------------
"""

        formatted_evidence.append(block)

    return "\n".join(formatted_evidence)


# BUILD EVIDENCE-BASED PROMPT

def build_prompt(
    query: str,
    evidence_registry: dict
) -> str:
    """
    Generate a strict evidence-based prompt.

    Gemini must return claims with references to
    retrieved chunk IDs.
    """

    evidence_context = format_evidence(
        evidence_registry
    )

    valid_ids = list(evidence_registry.keys())

    prompt = f"""
You are an evidence-based Multimodal PDF Question
Answering Assistant.

Your task is to answer questions using ONLY the
retrieved evidence from the uploaded PDF.

====================================================
USER QUESTION
====================================================

{query}


====================================================
RETRIEVED PDF EVIDENCE
====================================================

{evidence_context}


====================================================
VALID EVIDENCE REFERENCES
====================================================

{json.dumps(valid_ids, indent=2)}


====================================================
STRICT INSTRUCTIONS
====================================================

1. Use ONLY the supplied PDF evidence.

2. Do not use external knowledge or assumptions.

3. Every factual claim must have at least one
   supporting evidence ID.

4. Only use evidence IDs from the VALID EVIDENCE
   REFERENCES section.

5. Never invent chunk IDs or page numbers.

6. Numerical values must be directly supported
   by retrieved evidence.

7. When interpreting graphs, explain only what
   can reasonably be established from the visual
   evidence provided.

8. If evidence from multiple sources is required,
   include all relevant evidence IDs.

9. If the retrieved evidence does not answer the
   question, set insufficient_information to true.

10. Never fill missing information using general
    knowledge.

11. Do not generate unsupported conclusions.

12. Keep each claim focused on one factual statement.

13. Evidence precision must determine which source
    is preferred for factual answers.

14. For exact numerical questions, prioritize
    exact_structured table evidence.

15. Use textual evidence for explanations and
    statements explicitly described in the document.

16. Use approximate_visual evidence primarily for
    visual trends, patterns, comparisons and
    observations.

17. Never replace an exact numerical table value
    with an approximate value interpreted from a graph.

18. If exact structured evidence and approximate
    visual evidence differ, use the exact structured
    value and do not treat the approximate visual
    estimate as an exact contradiction.

19. If two exact sources contain conflicting values,
    report the conflict instead of arbitrarily
    selecting one.

====================================================
RESPONSE FORMAT
====================================================

Return ONLY valid JSON.

Use this exact structure:

{{
    "claims": [
        {{
            "claim": "A factual statement supported by PDF evidence.",
            "evidence_ids": [
                "text_page_10_1"
            ]
        }}
    ],
    "insufficient_information": false,
    "missing_information": ""
}}

IMPORTANT:

- Every claim must contain evidence_ids.
- Every evidence ID must come from the valid list.
- If no reliable answer exists, return an empty claims
  array and set insufficient_information to true.
- Do not include markdown formatting.
- Do not include explanations outside JSON.
"""

    return prompt


# PARSE GEMINI RESPONSE

def parse_json_response(response_text: str) -> dict:
    """
    Parse Gemini's structured JSON response.
    """

    if not response_text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    response_text = response_text.strip()

    # Remove accidental markdown code fences.

    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    parsed = json.loads(response_text)

    if not isinstance(parsed, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    if not isinstance(parsed.get("claims"), list):
        raise ValueError(
            "Gemini response does not contain a valid claims list."
        )

    return parsed


# VALIDATE CLAIMS

def validate_claims(
    generated_data: dict,
    evidence_registry: dict
) -> dict:
    """
    Validate every claim against retrieved evidence.

    Invalid chunk references are removed.

    Claims without valid references are rejected.

    NOTE:
    This validates citation identity, not semantic
    entailment. A valid ID alone does not prove that
    the cited content supports the claim.
    """

    valid_claims = []

    rejected_claims = []

    valid_evidence_ids = set(
        evidence_registry.keys()
    )

    for claim_data in generated_data.get("claims", []):

        if not isinstance(claim_data, dict):
            rejected_claims.append(claim_data)
            continue

        claim_text = claim_data.get("claim", "")

        evidence_ids = claim_data.get(
            "evidence_ids",
            []
        )

        if not isinstance(claim_text, str):
            rejected_claims.append(claim_data)
            continue

        if not claim_text.strip():
            rejected_claims.append(claim_data)
            continue

        if not isinstance(evidence_ids, list):
            rejected_claims.append(claim_data)
            continue

        # Retain only references that were actually retrieved.

        valid_references = list(dict.fromkeys(
            evidence_id
            for evidence_id in evidence_ids
            if isinstance(evidence_id, str)
            and evidence_id in valid_evidence_ids
        ))

        if not valid_references:

            print(
                f"REJECTED CLAIM: {claim_text}"
            )

            rejected_claims.append(claim_data)
            continue

        valid_claims.append({

            "claim": claim_text,

            "evidence_ids": valid_references

        })

    insufficient = generated_data.get(
        "insufficient_information",
        False
    )

    if not isinstance(insufficient, bool):
        insufficient = True

    return {

        "claims": valid_claims,

        "rejected_claims": rejected_claims,

        "insufficient_information": insufficient,

        "missing_information": generated_data.get(
            "missing_information",
            ""
        )

    }

# SAVE AND LOAD VERIFICATION CHECKPOINTS

def save_pending_verification(
        generated_data,
        validated_data,
        evidence_registry,
        image_paths
):
    """
    Save generated claims and retrived evdence.
    
    This allows semnetic verification to resume
    wihout repeating retrival or answer generation.
    """

    PENDING_VERIFICATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "generated_data" : generated_data,
        "validated_data" : validated_data,
        "evidence_registry" : evidence_registry,
        "image_paths" : image_paths,
        "status" : "panding_verification"
    }

    with open(
        PENDING_VERIFICATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            checkpoint,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(
        f"\nVerification Checkpoint saved: \n"
        f"{PENDING_VERIFICATION_PATH}"
    )

def load_pending_verification():

    if not PENDING_VERIFICATION_PATH.exists():

        raise FileNotFoundError(
            "No pending verification checkpoint found. "
            "Run the complete pipeline first "
        )

    with open(
        PENDING_VERIFICATION_PATH,"r",encoding="utf-8"
    ) as file:
        checkpoint = json.laod(file)

    if checkpoint.get("status") != "pending_verification" :

        raise ValueError(
            "Checkpoint is not pending verification"
        )

    print("\nVerification Checkpoint loaded successfully. ")

    return checkpoint


# Verify Claims

def verify_claims(
    validated_data: dict,
    evidence_registry: dict,
    image_paths: list = None
) -> dict:

    print(
        "\n========== SEMANTIC VERIFICATION STARTED =========="
    )

    claims = validated_data["claims"]

    rejected_claims = validated_data[
        "rejected_claims"
    ].copy()

    image_paths = image_paths or []

    if not claims:

        return {
            **validated_data,
            "insufficient_information": True
        }

    # =================================================
    # 1. BUILD COMPLETE EVIDENCE DATA
    # =================================================

    all_retrieved_evidence = []

    for evidence_id, evidence in evidence_registry.items():

        all_retrieved_evidence.append({

            "evidence_id": evidence_id,

            "document": evidence["document_name"],

            "page": evidence["page_number"],

            "type": evidence["content_type"],

            "precision": evidence["precision"],

            "evidence_priority": evidence["evidence_priority"],

            "content": evidence["content"]

        })

    # =================================================
    # 2. BUILD CLAIM DATA
    # =================================================

    verification_claims = []

    used_evidence_ids = set()

    for index, claim_data in enumerate(
        claims,
        start=1
    ):

        cited_evidence = []

        for evidence_id in claim_data["evidence_ids"]:

            evidence = evidence_registry.get(
                evidence_id
            )

            if not evidence:
                continue

            used_evidence_ids.add(evidence_id)

            cited_evidence.append({

                "evidence_id": evidence_id,

                "page": evidence["page_number"],

                "type": evidence["content_type"],

                "precision": evidence["precision"],

                "evidence_priority": evidence["evidence_priority"],

                "content": evidence["content"]

            })

        verification_claims.append({

            "claim_index": index,

            "claim": claim_data["claim"],

            "cited_evidence": cited_evidence

        })

    # =================================================
    # 3. BUILD VERIFICATION PROMPT
    # =================================================

    verification_prompt = f"""
You are a strict evidence verification system.

Verify each claim using ONLY the supplied PDF evidence.

EVIDENCE:

{json.dumps(all_retrieved_evidence, ensure_ascii=False)}

CLAIMS:

{json.dumps(verification_claims, ensure_ascii=False)}

RULES:

1. Verify every claim against its cited evidence.
2. Check numerical values, years, units and context.
3. Check other retrieved evidence for contradictions.
4. SUPPORTED: Evidence directly supports the claim.
5. UNSUPPORTED: Evidence does not support the claim.
6. UNCERTAIN: Evidence is ambiguous or incomplete.
7. Never assume missing information.
8. Return exactly one result for every claim.
9. Do not provide explanations.
10. Exact structured table values have higher
   numerical precision than approximate visual
   descriptions.

11. Do not reject an exact table value merely because
    a graph provides a rounded or approximate value.

12. Check whether the cited evidence actually
    supports the specific claim.

13. A valid evidence ID alone does not establish
    that a claim is supported.

14. If evidence contains genuine conflicting exact
    values, mark the claim UNCERTAIN.

15. A visual trend may support a qualitative claim,
    but must not be treated as an exact numerical
    measurement.

16. For every claim, provide a short reason explaining
    why the evidence supports, does not support,
    or cannot establish the claim.

17. Keep verification reasons concise and specific
    to the cited evidence.

18. Return a reason for every claim, including
    UNSUPPORTED and UNCERTAIN claims.

RETURN ONLY JSON:

{{
    "results": [
        {{
            "claim_index": 1,
            "status": "SUPPORTED",
            "reason": "The cited table explicitly provides the projected 2030 capacity."
        }}
    ]
}}

Allowed statuses:
SUPPORTED, UNSUPPORTED, UNCERTAIN
"""

    # =================================================
    # 4. ATTACH RELEVANT IMAGES
    # =================================================

    image_map = {

        item.get("chunk_id"): item.get("image_path")

        for item in image_paths

        if item.get("chunk_id")
        and item.get("image_path")

    }

    content_parts = [

        types.Part.from_text(
            text=verification_prompt
        )

    ]

    images_added = 0

    for evidence_id, image_path in image_map.items():
        if images_added >= MAX_IMAGES:
            break

        if evidence_id not in evidence_registry:
            continue

        evidence = evidence_registry[evidence_id]

        if evidence["precision"] != "approximate_visual":
            continue

        if not image_path:
            continue

        image_part = load_image_part(image_path)

        if image_part:

            content_parts.append(
                types.Part.from_text(
                    text=(
                        f"Visual Evidence ID: {evidence_id}\n"
                        f"Page : {evidence['page_number']}\n"
                        "Use this image to inspect visual trends"
                        "and approximate values."
                    )
                )
            )

            content_parts.append(image_part)

            images_added += 1

            print("Verification image added: {evidence_id}")


    print(
        f"Claims to verify: {len(claims)}"
    )

    print(
        f"Evidence items: {len(all_retrieved_evidence)}"
    )

    print(
        f"Images attached: {images_added}"
    )

    # =================================================
    # 5. GEMINI VERIFICATION
    # =================================================

    try:

        response = generate_content_with_retries(

            contents=content_parts,

            config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=4096,
            response_mime_type="application/json"
        ),

            max_retries=3

        )

        if not response.text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        print(
            "\n========== VERIFICATION RESPONSE =========="
        )

        print(response.text)

        # Check whether Gemini completed generation.

        if response.candidates:

            finish_reason = (
                response.candidates[0].finish_reason
            )

            print(
                "Finish reason:",
                finish_reason
            )

            if str(finish_reason).upper().endswith(
                "MAX_TOKENS"
            ):

                raise ValueError(
                    "Verification response was truncated "
                    "because MAX_TOKENS was reached."
                )

        # Parse JSON.

        result = json.loads(
            response.text.strip()
        )

        if not isinstance(result, dict):

            raise ValueError(
                "Invalid verification JSON."
            )

        if not isinstance(
            result.get("results"),
            list
        ):

            raise ValueError(
                "Missing verification results."
            )

    except Exception as error:

        print(
            f"\nVERIFICATION ERROR: {error}"
        )

        print(
            "Failing closed: no claims will be accepted."
        )

        for claim in claims:

            rejected_claims.append({

                **claim,

                "verification": "UNCERTAIN",

                "reason": (
                    "Semantic verification could not "
                    "be completed."
                )

            })

        return {

            "claims": [],

            "rejected_claims": rejected_claims,

            "insufficient_information": True,

            "missing_information": (
                "Semantic verification failed. "
                "The generated claims could not be verified."
            )

        }

    # =================================================
    # 6. VALIDATE VERIFICATION RESULTS
    # =================================================

    verification_results = {}

    for item in result["results"]:

        if not isinstance(item, dict):
            continue

        claim_index = item.get(
            "claim_index"
        )

        if (
            isinstance(claim_index, int)
            and not isinstance(claim_index, bool)
            and 1 <= claim_index <= len(claims)
            and claim_index not in verification_results
        ):

            verification_results[
                claim_index
            ] = item

    # =================================================
    # 7. ACCEPT OR REJECT CLAIMS
    # =================================================

    verified_claims = []

    for index, claim_data in enumerate(
        claims,
        start=1
    ):

        verification = verification_results.get(
            index
        )

        if verification is None:

            verdict = "UNCERTAIN"

            reason = (
                "Gemini did not return a result for this claim."
            )

        else:

            verdict = str(
                verification.get(
                    "status",
                    "UNCERTAIN"
                )
            ).upper()

            reason = verification.get(
                "reason",
                "No verification explanation."
            )

            if verdict not in [
                "SUPPORTED",
                "UNSUPPORTED",
                "UNCERTAIN"
            ]:

                verdict = "UNCERTAIN"

            if verification.get(
                "conflict_detected",
                False
            ):

                verdict = "UNCERTAIN"

                reason = (
                    "Conflicting evidence detected. "
                    + str(reason)
                )

        print(
            f"\nClaim {index}: {claim_data['claim']}"
        )

        print(
            f"Verdict: {verdict}"
        )

        print(
            f"Reason: {reason}"
        )

        if verdict == "SUPPORTED":

            verified_claims.append(
                claim_data
            )

        else:

            rejected_claims.append({

                **claim_data,

                "verification": verdict,

                "reason": reason

            })

    # =================================================
    # 8. FINAL SUMMARY
    # =================================================

    print(
        "\n========== VERIFICATION COMPLETED =========="
    )

    print(
        "Supported:",
        len(verified_claims)
    )

    print(
        "Rejected:",
        len(rejected_claims)
    )

    return {

        "claims": verified_claims,

        "rejected_claims": rejected_claims,

        "insufficient_information": (

            validated_data["insufficient_information"]

            or len(verified_claims) == 0

            or len(verified_claims) < len(claims)

        ),

        "missing_information": validated_data.get(
            "missing_information",
            ""
        )

    }


# FORMAT FINAL ANSWER

def format_final_answer(
    validated_data: dict,
    evidence_registry: dict
) -> str:
    """
    Format the validated claims with source references.
    """

    claims = validated_data["claims"]

    if not claims:

        return (
            "I could not establish a reliable answer "
            "from the retrieved PDF evidence. "
            "The available document content does not "
            "provide sufficient supporting information."
        )

    answer_parts = []

    used_evidence_ids = set()

    answer_parts.append("## Answer\n")

    for index, claim_data in enumerate(claims, start=1):

        claim_text = claim_data["claim"]

        evidence_ids = claim_data["evidence_ids"]

        used_evidence_ids.update(evidence_ids)

        references = ", ".join(
            f"[{evidence_id}]"
            for evidence_id in evidence_ids
        )

        answer_parts.append(
            f"{index}. {claim_text} {references}\n"
        )

    # SOURCE REFERENCES

    answer_parts.append(
        "\n## Source References\n"
    )

    for evidence_id in sorted(used_evidence_ids):

        evidence = evidence_registry[evidence_id]

        answer_parts.append(
            f"""
- **Chunk ID:** {evidence_id}
  - **Document:** {evidence['document_name']}
  - **Page:** {evidence['page_number']}
  - **Content Type:** {evidence['content_type']}
"""
        )

    # ---------------------------------------------
    # INSUFFICIENT INFORMATION
    # ---------------------------------------------

    if validated_data["insufficient_information"]:

        missing = validated_data.get(
            "missing_information",
            ""
        )

        answer_parts.append(
            "\n## Additional Information\n"
        )

        answer_parts.append(
            missing.strip()
            if isinstance(missing, str) and missing.strip()
            else (
                "Some requested information could not "
                "be established from the retrieved evidence."
            )
        )

    # VALIDATION NOTICE

    rejected_count = len(
        validated_data["rejected_claims"]
    )

    if rejected_count:

        answer_parts.append(
            "\n\n## Verification Notice\n"
        )

        answer_parts.append(
            f"{rejected_count} claims were excluded "
            "because their supporting evidence could "
            "not be verified."
        )

    return "\n".join(answer_parts)


# GENERATE MULTIMODAL ANSWER

def generate_answer(
    query: str,
    top_k: int = 3,
    max_images: int = MAX_IMAGES
):
    """
    Retrieve PDF evidence and generate a traceable
    multimodal answer using Gemini.
    """

    print(
        "\n========== MULTIMODAL RAG STARTED =========="
    )

    # 1. RETRIEVE RELEVANT CHUNKS

    print("\n[1] Retrieving relevant chunks...")

    retrieved_chunks = retrieve_by_query(
        query,
        top_k=top_k
    )

    if not retrieved_chunks:

        return (
            "I could not find relevant information "
            "in the uploaded PDF."
        )

    print(
        f"Retrieved chunks: {len(retrieved_chunks)}"
    )

    # 2. BUILD MULTIMODAL CONTEXT

    print("\n[2] Building multimodal context...")

    context = build_multimodal_context(
        retrieved_chunks
    )

    print(
        f"Text chunks: {len(context.get('text_chunks', []))}"
    )

    print(
        f"Table chunks: {len(context.get('table_chunks', []))}"
    )

    print(
        f"Visual chunks: {len(context.get('visual_chunks', []))}"
    )

    # ---------------------------------------------
    # 3. BUILD EVIDENCE REGISTRY
    # ---------------------------------------------

    print("\n[3] Building evidence registry...")

    evidence_registry = build_evidence_registry(
        retrieved_chunks
    )

    if not evidence_registry:

        return (
            "Retrieved content does not contain "
            "traceable evidence references."
        )
    
    inspect_evidence_registry(evidence_registry)

    # 4. BUILD PROMPT

    print("\n[4] Preparing evidence-based prompt...")

    prompt = build_prompt(
        query,
        evidence_registry
    )

    # 5. LOAD RETRIEVED IMAGES

    print("\n[5] Loading retrieved images...")

    image_paths = context.get(
        "image_paths",
        []
    )

    content_parts = [

        types.Part.from_text(
            text=prompt
        )

    ]

    images_added = 0

    for image_info in image_paths:

        if images_added >= max_images:
            break

        image_path = image_info.get(
            "image_path"
        )

        chunk_id = image_info.get(
            "chunk_id"
        )

        # Only send images associated with
        # registered evidence.

        if chunk_id not in evidence_registry:
            continue

        if not image_path:
            continue

        image_part = load_image_part(
            image_path
        )

        if image_part is not None:

            # Identify the source before attaching
            # the actual image.

            content_parts.append(
                types.Part.from_text(
                    text=(
                        f"\nIMAGE EVIDENCE:\n"
                        f"Chunk ID: {chunk_id}\n"
                        f"Page: "
                        f"{evidence_registry[chunk_id]['page_number']}\n"
                    )
                )
            )

            content_parts.append(
                image_part
            )

            images_added += 1

            print(
                f"Image added: {chunk_id}"
            )

    print(
        f"Total images sent to Gemini: {images_added}"
    )

    # 6. GENERATE STRUCTURED ANSWER

    print("\n[6] Generating answer with Gemini...")

    try:

        # ---------------------------------------------
        # GEMINI GENERATION WITH RETRIES
        # ---------------------------------------------

        response = generate_content_with_retries(

            contents=content_parts,

            config=types.GenerateContentConfig(

                temperature=0.1,

                max_output_tokens=4096,

                response_mime_type="application/json"

            ),

            max_retries=3

        )

        # ---------------------------------------------
        # CHECK GENERATION COMPLETION
        # ---------------------------------------------

        if response.candidates:

            finish_reason = (
                response.candidates[0].finish_reason
            )

            print(
                "Generation finish reason:",
                finish_reason
            )

            if str(finish_reason).upper().endswith(
                "MAX_TOKENS"
            ):

                raise ValueError(
                    "Answer generation was truncated "
                    "because MAX_TOKENS was reached."
                )

        # ---------------------------------------------
        # CHECK RESPONSE
        # ---------------------------------------------

        if not response.text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        # ---------------------------------------------
        # PARSE GEMINI RESPONSE
        # ---------------------------------------------

        print(
            "\n[7] Parsing Gemini response..."
        )

        generated_data = parse_json_response(
            response.text
        )

        if response.candidates:

            finish_reason = (
                response.candidates[0].finish_reason
            )

            print(
                "Generation finish reason:",
                finish_reason
            )

            if str(finish_reason).upper().endswith(
                "MAX_TOKENS"
            ):

                raise ValueError(
                    "Answer generation was truncated "
                    "because MAX_TOKENS was reached."
                )

        # ---------------------------------------------
        # VALIDATE CITATIONS
        # ---------------------------------------------

        print(
            "\n[8] Validating evidence references..."
        )

        validated_data = validate_claims(
            generated_data,
            evidence_registry
        )

        # ---------------------------------------------
        # SAVE VERIFICATION CHECKPOINT
        # ---------------------------------------------

        print(
            "\n[9] Saving verification checkpoint..."
        )

        save_pending_verification(
            generated_data=generated_data,
            validated_data=validated_data,
            evidence_registry=evidence_registry,
            image_paths=image_paths
        )

        # ---------------------------------------------
        # SEMANTIC VERIFICATION
        # ---------------------------------------------

        print(
            "\n[10] Verifying generated claims..."
        )

        validated_data = verify_claims(
            validated_data,
            evidence_registry,
            image_paths=image_paths
        )

        print(
            f"Verified claims: {len(validated_data['claims'])}"
        )

        print(
            f"Rejected claims: {len(validated_data['rejected_claims'])}"
        )

        # ---------------------------------------------
        # FORMAT FINAL ANSWER
        # ---------------------------------------------

        print(
            "\n[10] Formatting final answer..."
        )

        final_answer = format_final_answer(
            validated_data,
            evidence_registry
        )

        print(
            "\n========== ANSWER GENERATED ==========\n"
        )

        return final_answer

    except Exception as error:

        print(
            f"\nGemini generation failed: {error}"
        )

        return (
            "Unable to generate a traceable answer. "
            "Please check the generation logs."
        )

    except Exception as error:

        print(
            f"\nGemini generation failed: {error}"
        )

        return (
            "Unable to generate a traceable answer. "
            "Please check the generation logs."
        )

# RESUME PENDING VERIFICATION

def resume_pending_verification():

    print("=== RESUMING PENDING VERIFICATION ===")

    checkpoint = load_pending_verification
    validated_data = checkpoint["validated_data"]
    evidence_registry = checkpoint["evidence_registry"]
    image_paths = checkpoint["image_paths"]

    print(
        f"Claims waiting for verification: "
        f"{len(validated_data['claims'])}"
    )

    print(
        f"Available evidence: "
        f"{len(evidence_registry)}"
    )


    verified_data = verify_claims(
        validated_data,
        evidence_registry,
        image_paths=image_paths
    )

    if verified_data["insufficient_information"]:

        print("\nVerification could not be completed.")

        print("Checkpoint preserved for another attmept.")

        return format_final_answer(
            verified_data,
            evidence_registry
        )

    result_path = (
        PENDING_VERIFICATION_PATH.parent
        / "verified_answer.json"
    )

    with open(
        result_path,"w",encoding="utf-8"
    ) as file :
        json.dump(
            verified_data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    checkpoint["status"] = "completed"

    checkpoint["verified_data"] = verified_data

    with open(
        PENDING_VERIFICATION_PATH,"w",encoding="utf-8"
    ) as file:
        json.dump(
            checkpoint,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print("\nVerification Completed Successfully..")

    print(f"Verified result saved to {result_path}")

    final_answer = format_final_answer(
        verified_data,
        evidence_registry
    )

    return final_answer



# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verify-pending",
        action="store_true",
        help="Resume semantic verification from saved claims."
    )

    args = parser.parse_args()

    if args.verify_pending:

        answer = resume_pending_verification()

    else:

        test_query = (
            "What are the global renewable energy "
            "capacity projections for 2030?"
        )

        answer = generate_answer(
            query=test_query,
            top_k=3,
            max_images=4
        )

    print(
        "\n========== FINAL ANSWER ==========\n"
    )

    print(answer)