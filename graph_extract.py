from pathlib import Path
import json
import re
from collections import defaultdict
import constant

CHUNKS_FILE = constant.CHUNKS_FILE
ENTITIES_FILE = constant.ENTITIES_FILE
RELATIONS_FILE = constant.RELATIONS_FILE




# Utility functions


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_entity_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s\-]", "", name)

    if name in constant.ALIASES:
        name = constant.ALIASES[name]

    return name


def find_term_mentions(text: str, term: str) -> bool:
    """
    Match full term with word boundaries.
    """
    escaped = re.escape(term)
    pattern = rf"\b{escaped}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def infer_entity_type(entity_name: str) -> str:
    for entity_type, term_list in constant.DOMAIN_TERMS.items():
        for term in term_list:
            if normalise_entity_name(term) == entity_name:
                return entity_type
    return "unknown"


# 3. Entity extraction


def extract_entities_from_chunk(chunk: dict) -> list[dict]:
    text = chunk["text"]
    text_lower = text.lower()

    found = {}

    # 3a. Dictionary-based extraction
    for entity_type, term_list in constant.DOMAIN_TERMS.items():
        for term in term_list:
            if find_term_mentions(text_lower, term.lower()):
                canonical = normalise_entity_name(term)
                found[canonical] = {
                    "entity_name": canonical,
                    "entity_type": entity_type,
                }

    # 3b. Very simple noun-phrase-like pattern for title case / acronym phrases
    # This helps catch things like "World Health Organization"
    phrase_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b"
    extra_phrases = re.findall(phrase_pattern, text)

    for phrase in extra_phrases:
        canonical = normalise_entity_name(phrase)
        if len(canonical) > 2 and canonical not in found:
            found[canonical] = {
                "entity_name": canonical,
                "entity_type": infer_entity_type(canonical),
            }

    entities = []
    for entity_name, info in found.items():
        entities.append({
            "entity_id": f"ent::{entity_name.replace(' ', '_')}",
            "canonical_name": entity_name,
            "entity_type": info["entity_type"],
            "aliases": [],
            "source_chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page": chunk["page"],
            "section": chunk.get("section"),
        })

    return entities


# -----------------------------------
# 4. Relation extraction
# -----------------------------------

def extract_relations_from_chunk(chunk: dict, chunk_entities: list[dict]) -> list[dict]:
    """
    MVP relation strategy:
    If 2 entities co-occur in the same chunk, create a 'co_occurs_with' relation.
    """
    relations = []
    entity_names = sorted({e["canonical_name"] for e in chunk_entities})

    for i in range(len(entity_names)):
        for j in range(i + 1, len(entity_names)):
            subj = entity_names[i]
            obj = entity_names[j]

            relations.append({
                "relation_id": f"rel::{subj.replace(' ', '_')}::co_occurs_with::{obj.replace(' ', '_')}::{chunk['chunk_id']}",
                "subject_id": f"ent::{subj.replace(' ', '_')}",
                "subject": subj,
                "predicate": "co_occurs_with",
                "object_id": f"ent::{obj.replace(' ', '_')}",
                "object": obj,
                "source_chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "confidence": 0.5,
                "method": "chunk_cooccurrence",
            })

    return relations


# -----------------------------------
# 5. Deduplicate / merge entities
# -----------------------------------

def merge_entities(entity_records: list[dict]) -> list[dict]:
    merged = {}

    for ent in entity_records:
        entity_id = ent["entity_id"]

        if entity_id not in merged:
            merged[entity_id] = {
                "entity_id": entity_id,
                "canonical_name": ent["canonical_name"],
                "entity_type": ent["entity_type"],
                "aliases": list(set(ent.get("aliases", []))),
                "source_chunk_ids": [ent["source_chunk_id"]],
                "document_ids": [ent["document_id"]],
                "pages": [ent["page"]],
                "sections": [ent["section"]] if ent.get("section") else [],
            }
        else:
            merged[entity_id]["source_chunk_ids"].append(ent["source_chunk_id"])
            merged[entity_id]["document_ids"].append(ent["document_id"])
            merged[entity_id]["pages"].append(ent["page"])
            if ent.get("section"):
                merged[entity_id]["sections"].append(ent["section"])

    # unique / sorted cleanup
    for entity_id in merged:
        merged[entity_id]["source_chunk_ids"] = sorted(set(merged[entity_id]["source_chunk_ids"]))
        merged[entity_id]["document_ids"] = sorted(set(merged[entity_id]["document_ids"]))
        merged[entity_id]["pages"] = sorted(set(merged[entity_id]["pages"]))
        merged[entity_id]["sections"] = sorted(set(merged[entity_id]["sections"]))

    return list(merged.values())


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for rel in relations:
        key = (
            rel["subject_id"],
            rel["predicate"],
            rel["object_id"],
            rel["source_chunk_id"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(rel)

    return deduped


# -----------------------------------
# 6. Main pipeline
# -----------------------------------

def main():
    chunks = load_chunks(CHUNKS_FILE)

    all_entity_mentions = []
    all_relations = []

    for chunk in chunks:
        chunk_entities = extract_entities_from_chunk(chunk)
        chunk_relations = extract_relations_from_chunk(chunk, chunk_entities)

        all_entity_mentions.extend(chunk_entities)
        all_relations.extend(chunk_relations)

    merged_entities = merge_entities(all_entity_mentions)
    deduped_relations = deduplicate_relations(all_relations)

    ENTITIES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_entities, f, indent=2, ensure_ascii=False)

    with open(RELATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped_relations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(merged_entities)} entities -> {ENTITIES_FILE}")
    print(f"Saved {len(deduped_relations)} relations -> {RELATIONS_FILE}")

    if merged_entities:
        print("\nExample entity:")
        print(json.dumps(merged_entities[0], indent=2, ensure_ascii=False))

    if deduped_relations:
        print("\nExample relation:")
        print(json.dumps(deduped_relations[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()