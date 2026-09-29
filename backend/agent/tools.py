"""
Function-calling tool definitions the Groq model can choose to call, plus
their Python implementations.

The tools are responsible for:
- Searching Hindsight memory
- Checking whether an entity actually exists in the indexed codebase
- Returning CONFIRMED / INFERRED / STALE status
- Recording human validation only for real code entities
"""

from . import staleness


# ============================================================
# TOOL SCHEMAS
# ============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "recall_codebase",
            "description": (
                "Search Hindsight's memory for facts, observations, or mental "
                "models relevant to a question about the legacy codebase. "
                "Use this before answering any question about what code does. "
                "Only treat information retrieved from the indexed codebase "
                "as evidence about the actual code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": (
                            "Optional specific class or method names to "
                            "narrow the search."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_entity_confidence",
            "description": (
                "Check whether a class or method actually exists in the "
                "indexed legacy codebase and, if it exists, return its "
                "confidence state. The possible states are CONFIRMED "
                "(human-validated and code unchanged), INFERRED "
                "(found in the indexed codebase but never human-validated), "
                "or STALE (human-validated but code changed afterward). "
                "If the entity does not exist in the indexed codebase, "
                "return UNKNOWN. Never assume an entity is real merely "
                "because the user mentioned its name. Never create a new "
                "entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": (
                            "Exact class or method name from the indexed "
                            "codebase."
                        ),
                    },
                },
                "required": ["entity"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "record_validation",
            "description": (
                "Record that a human developer or BA confirmed or corrected "
                "an explanation of an entity that already exists in the "
                "indexed codebase. Do not use this tool for entities that "
                "do not exist in the indexed codebase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                    },
                    "correction": {
                        "type": "string",
                        "description": (
                            "The confirmed or corrected explanation, in full."
                        ),
                    },
                    "validated_by": {
                        "type": "string",
                        "description": (
                            "Name of the person confirming the explanation."
                        ),
                    },
                },
                "required": [
                    "entity",
                    "correction",
                    "validated_by",
                ],
            },
        },
    },
]


# ============================================================
# MEMORY RESPONSE LIMITS
# ============================================================

_MAX_CONTENT_CHARS = 500


# ============================================================
# HINDSIGHT RESPONSE CLEANUP
# ============================================================

def _slim_memory(item: dict) -> dict:
    """
    Keep only the information the model actually needs.

    Drop embeddings, vectors, internal IDs, timestamps, scores, etc.
    """

    if not isinstance(item, dict):
        return item

    content = item.get("content") or item.get("text") or ""

    if isinstance(content, str) and len(content) > _MAX_CONTENT_CHARS:
        content = (
            content[:_MAX_CONTENT_CHARS]
            + "... [truncated]"
        )

    entities = item.get("entities")

    slim = {
        "content": content,
    }

    if entities:
        slim["entities"] = [
            e.get("text")
            if isinstance(e, dict)
            else e
            for e in entities
        ]

    return slim


def _slim_recall_result(
    result: dict,
    max_items: int = 3,
) -> dict:
    """
    Strip Hindsight's raw recall response down to a small,
    LLM-friendly structure.
    """

    if not isinstance(result, dict):
        return {
            "memories": []
        }

    items = (
        result.get("memories")
        or result.get("items")
        or result.get("results")
        or []
    )

    slim_items = [
        _slim_memory(item)
        for item in items[:max_items]
        if isinstance(item, dict)
    ]

    return {
        "memories": slim_items
    }


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def make_tool_impls(hindsight_client):
    """
    Bind tool implementations to a live HindsightClient instance.
    """

    # --------------------------------------------------------
    # RECALL CODEBASE
    # --------------------------------------------------------

    def recall_codebase(
        query: str,
        entities: list[str] = None,
    ) -> dict:

        result = hindsight_client.recall(
            query=query,
            entities=entities,
            top_k=3,
        )

        return _slim_recall_result(result)


    # --------------------------------------------------------
    # GET ENTITY CONFIDENCE
    # --------------------------------------------------------

    def get_entity_confidence(
        entity: str,
    ) -> dict:

        # IMPORTANT:
        # Never allow an arbitrary user/model-generated entity
        # to become part of Codebase Memory.

        if not staleness.entity_exists(entity):

            return {
                "entity": entity,
                "badge": "UNKNOWN",
                "exists": False,
                "message": (
                    f"'{entity}' was not found in the indexed "
                    "codebase. Do not treat this as a real "
                    "class or method."
                ),
            }

        status = staleness.status(entity)

        return {
            "entity": status.entity,
            "badge": status.badge,
            "exists": True,
            "last_validated_at": status.last_validated_at,
            "validated_by": status.validated_by,
            "last_code_changed_at": status.last_code_changed_at,
        }


    # --------------------------------------------------------
    # RECORD VALIDATION
    # --------------------------------------------------------

    def record_validation(
        entity: str,
        correction: str,
        validated_by: str,
    ) -> dict:

        # IMPORTANT:
        # Do not allow validation of entities that aren't
        # actually present in the indexed codebase.

        if not staleness.entity_exists(entity):

            return {
                "status": "rejected",
                "entity": entity,
                "badge": "UNKNOWN",
                "message": (
                    f"'{entity}' does not exist in the indexed "
                    "codebase. Validation was not recorded."
                ),
            }

        hindsight_client.retain(
            content=correction,
            entities=[entity],
            kind="world",
            metadata={
                "validated_by": validated_by,
                "type": "human_validation",
            },
        )

        staleness.mark_validated(
            entity,
            validated_by,
        )

        return {
            "status": "recorded",
            "entity": entity,
            "badge": "CONFIRMED",
        }


    # --------------------------------------------------------
    # RETURN TOOL IMPLEMENTATIONS
    # --------------------------------------------------------

    return {
        "recall_codebase": recall_codebase,
        "get_entity_confidence": get_entity_confidence,
        "record_validation": record_validation,
    }
