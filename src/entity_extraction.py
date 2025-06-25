import asyncio
import os
import json
import re
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")  # if needed
)

model_name = os.environ.get("OPENAI_MODEL", "gpt-4")
semaphore = asyncio.Semaphore(5)

MAX_CHUNK_LENGTH = 6000
MAX_HISTORY_NAMES = 100  # how many prior names to pass for dedupe-awareness

# Entity types and their extra, type-specific prompt instructions
TYPE_INSTRUCTIONS: Dict[str, str] = {
    "Person": (
        "If there are any aliases for a Person, use '@' like 'Ram @ Raju'. "
        "Strictly do not write 'r/o' (resident of) or other address snippets in the name. "
        "Extract family members (Father/Mother/Spouse) of the main entity only as separate entity, if explicitly mentioned."
    ),
    "Organization": (
        "Gangs or groups count as Organizations here. "
        "Include any named gangs, unions, NGOs, political parties etc. "
        "Do not classify geographic areas (e.g. 'Delhi') as a Gang."
    ),
    "Location": (
        "Include any geographic place: villages, cities, states, landmarks. "
        "Do not include police stations, organizations, or educational institutions here."
    ),
    "Police Station": (
        "Only include formal police station names (e.g. 'P.S Shivaji Colony')."
    ),
    "Criminal Activity": (
        "Name discrete criminal acts or events (e.g. 'Attack on Sunny', 'Firing at Pandit RM Hospital')."
    ),
    "Date": (
        "Include exact dates or date ranges (e.g. '29-09-2022', 'March–April 2023')."
    ),
    "Educational Institution": (
        "Schools, colleges, training centers. "
        "Do not include generic terms like 'School' or 'College' without a specific name."
    ),
    "Vehicle": (
        "Each vehicle as a separate entry (e.g. 'Santro car', 'Motorcycle'). "
        "Do not group them under generic 'Car'."
    ),
}

# JSON schema for validation
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "relationship_to_main": {"type": "string"}
                },
                "required": ["name", "type", "relationship_to_main"]
            }
        }
    },
    "required": ["entities"]
}

async def enrich_relationships(
    chunk_text: str,
    names: List[str],
    main_entity: str,
    system_prompt: str
) -> Dict[str, str]:
    """
    Given a list of names and a chunk of text, ask the LLM to return relationships of each name
    to the main entity. Returns a mapping: name -> relationship string.
    """
    try:
        # Construct input prompt
        query = (
            f"The following names were extracted from a legal/crime document about '{main_entity}':\n"
            f"{', '.join(names)}\n\n"
            "For each name, briefly describe their relationship to the main entity based on this context:\n"
            f"{chunk_text}\n\n"
            "Return the result strictly as JSON:\n"
            "{\n"
            "  \"Name1\": \"relationship_to_main\",\n"
            "  \"Name2\": \"relationship_to_main\"\n"
            "}"
        )

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        raw_content = response.choices[0].message.content.strip()
        # Extract valid JSON from response
        content = raw_content
        if content.startswith("```"):
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and part.endswith("}"):
                    content = part
                    break
        return json.loads(content)
    except Exception as e:
        print(f"[DEBUG] Relationship enrichment failed: {e}")
        return {name: "Unknown relationship" for name in names}

async def retry_extraction(
    chunk_text: str,
    entity_type: str,
    system_prompt: str,
    main_entity: str = ""
) -> Dict[str, Any]:
    """
    Fallback: extract names, then enrich their relationship using a second LLM step.
    """
    try:
        simplified_prompt = (
            f"Extract ONLY the names of {entity_type} entities from the text, separated by commas. "
            "Do NOT include any JSON formatting, markdown, or code fences—just a plain comma-separated list of names. "
            "Use '@' if an alias is known (e.g. 'MainName @ Alias')."
        )

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": simplified_prompt + "\n\n" + chunk_text}
            ]
        )
        content = response.choices[0].message.content.strip()
        names = [name.strip() for name in content.split(",") if name.strip()]
        names = [n for n in names if not re.search(r"[{}\[\]```]", n)]

        # Enrich with relationship info
        relationship_map = await enrich_relationships(chunk_text, names, main_entity, system_prompt)

        return {
            "entities": [
                {
                    "name": name,
                    "type": entity_type,
                    "relationship_to_main": relationship_map.get(name, "Unknown relationship")
                }
                for name in names
            ]
        }
    except Exception as e:
        return {"error": f"Retry failed: {str(e)}", "entities": []}

async def extract_entities_from_chunk_by_type(
    chunk_text: str,
    system_prompt: str,
    entity_type: str,
    seen_names: List[str],
    input_prompt: str = "",
    main_entity: str = "",
    temperature=0.3
) -> Dict[str, Any]:
    async with semaphore:
        # Build history snippet
        history = ""
        if seen_names:
            recent = seen_names[-MAX_HISTORY_NAMES:]
            history = "Previously extracted:\n" + ", ".join(recent) + "\n"
        if main_entity:
            history += f"Interrogation for main entity: {main_entity}\n"
        # Inject type-specific guidance
        type_instr = TYPE_INSTRUCTIONS.get(entity_type, "")
        # Combine all parts
        instruction = (
            f"{history}"
            f"Extract only **{entity_type}** entities from the text.\n"
            f"{type_instr}\n"
            f"{input_prompt}\n"
            "If an entity exactly matches a prior name or alias, do NOT repeat it.\n"
            "But if the prior extraction was incomplete, you can repeat it with more details; such as adding an alias.\n"
            "Dont write any 'r/o' in names.\n"
            "Return STRICTLY a JSON object with schema:\n"
            "{\n"
            "  \"entities\": [ { \"name\": \"...\", \"type\": \"...\", \"relationship_to_main\": \"...\" } ]\n"
            "}\n"
            "No markdown or extra text."
        )

        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction + "\n\n" + chunk_text}
                ],
                temperature=temperature
            )
            raw = resp.choices[0].message.content.strip()
            # strip code fences
            if raw.startswith("```"):
                parts = raw.split("```")
                for p in parts:
                    if p.strip().startswith("{"):
                        raw = p.strip()
                        break
                else:
                    raw = re.sub(r"^```(?:json)?", "", raw)
                    raw = re.sub(r"```$", "", raw).strip()
            data = json.loads(raw)
            validate(instance=data, schema=ENTITY_SCHEMA)
            return data
        except Exception as e:
            print(f"[DEBUG] Failed JSON parse for {entity_type}: {e}\n{raw}")
            return await retry_extraction(chunk_text, entity_type, system_prompt, main_entity)



def split_into_chunks(texts: List[str], max_length: int) -> List[List[str]]:
    chunks, current, length = [], [], 0
    for t in texts:
        l = len(t)
        if current and length + l > max_length:
            chunks.append(current)
            current, length = [t], l
        else:
            current.append(t)
            length += l
    if current: chunks.append(current)
    return chunks

def strip_address_fragment(s: str) -> str:
    """
    Remove trailing 'r/o ...' or similar address fragments from a name segment.
    E.g. 'Kalu r/o Chichdana' -> 'Kalu'
    """
    # Look for patterns like ' r/o ' or ' R/O ' (case-insensitive)
    # We split at the first occurrence of ' r/o ' (case-insensitive).
    parts = re.split(r"\s+r/?o\s+", s, flags=re.IGNORECASE)
    # Take the first part as the cleaned name
    return parts[0].strip()

def normalize_alias_format(name: str) -> Tuple[str, str]:
    """
    Detect alias patterns and strip any 'r/o ...' fragments.
    Returns (main_name, alias) or (name, "") if no alias found.
    """
    name = name.strip()
    # First remove surrounding quotes/backticks if any
    # (optional, depending on extraction noise)
    # Detect patterns:
    m = re.match(r"^(.*?)\s*\(\s*alias\s+(.+?)\s*\)$", name, flags=re.IGNORECASE)
    if m:
        main_raw = m.group(1).strip()
        alias_raw = m.group(2).strip()
        # Strip address fragments
        main = strip_address_fragment(main_raw)
        alias = strip_address_fragment(alias_raw)
        return main, alias
    m2 = re.match(r"^(.*?)\s*@\s*(.+)$", name)
    if m2:
        main_raw = m2.group(1).strip()
        alias_raw = m2.group(2).strip()
        main = strip_address_fragment(main_raw)
        alias = strip_address_fragment(alias_raw)
        return main, alias
    # No alias pattern; also strip address from standalone name
    cleaned = strip_address_fragment(name)
    return cleaned, ""

def postprocess_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize alias format, strip 'r/o ...', then merge:
    - If “Main @ Alias” exists, any standalone “Alias” is merged under that.
    - If “Alias” appears first, it is merged once the “Main @ Alias” appears.
    - Deduplicate exact duplicates and combine relationship fields.
    """
    # First pass: normalize each entity's name into canonical form
    canonical_entities: List[Dict[str, Any]] = []
    # Maps: alias_lower -> canonical_name, main_lower -> canonical_name (when alias exists)
    alias_to_canonical: Dict[str, str] = {}
    main_to_canonical: Dict[str, str] = {}
    # Also keep track of relationship texts to merge later
    for e in entities:
        raw_name = e.get("name", "").strip()
        main, alias = normalize_alias_format(raw_name)
        if alias:
            canonical_name = f"{main} @ {alias}"
            # remember mapping
            alias_lower = alias.lower()
            main_lower = main.lower()
            alias_to_canonical[alias_lower] = canonical_name
            main_to_canonical[main_lower] = canonical_name
        else:
            canonical_name = main
        # Store normalized entity
        canonical_entities.append({
            "name": canonical_name,
            "type": e.get("type", "").strip(),
            "relationship_to_main": e.get("relationship_to_main", "").strip()
        })

    # Second pass: merge entities
    merged: Dict[str, Dict[str, Any]] = {}
    for e in canonical_entities:
        name = e["name"]
        # Determine target canonical name:
        # 1. If this name matches exactly a canonical key, use it.
        # 2. Else if name.lower() is in alias_to_canonical, map to that.
        # 3. Else if name.lower() is in main_to_canonical, map to that.
        lower = name.lower()
        if name in merged:
            target = name
        elif lower in alias_to_canonical:
            target = alias_to_canonical[lower]
        elif lower in main_to_canonical:
            target = main_to_canonical[lower]
        else:
            target = name

        # Initialize or merge
        if target not in merged:
            # If e["name"] != target, we may want to adjust type. But usually type same.
            merged[target] = {
                "name": target,
                "type": e["type"],
                "relationship_to_main": e["relationship_to_main"]
            }
        else:
            # Merge relationship_to_main if new info
            existing_rel = merged[target]["relationship_to_main"]
            new_rel = e["relationship_to_main"]
            if new_rel and new_rel not in existing_rel:
                if existing_rel:
                    merged[target]["relationship_to_main"] = existing_rel + " | " + new_rel
                else:
                    merged[target]["relationship_to_main"] = new_rel
            # Optionally, you could reconcile types if needed

    # Return as list
    return list(merged.values())


async def hierarchical_extract_entities(
    paragraphs: List[str],
    system_prompt: str,
    input_prompt: str = "",
    main_entity: str = "",
    temperature=0.3
) -> Dict[str, Any]:
    chunk_groups = split_into_chunks(paragraphs, MAX_CHUNK_LENGTH)
    seen_names: List[str] = []
    accumulated: List[Dict[str, Any]] = []
    errors: List[str] = []

    for group in chunk_groups:
        text = "\n\n".join(group)
        for et in TYPE_INSTRUCTIONS.keys():
            res = await extract_entities_from_chunk_by_type(
                chunk_text=text,
                system_prompt=system_prompt,
                entity_type=et,
                seen_names=seen_names,
                input_prompt=input_prompt,
                main_entity=main_entity,
                temperature=temperature
            )
            if not isinstance(res, dict):
                continue
            if "error" in res:
                errors.append(res["error"])
                continue
            batch = postprocess_entities(res.get("entities", []))
            combined = postprocess_entities(accumulated + batch)
            accumulated = combined
            seen_names = [e["name"] for e in accumulated]

    output = {"entities": accumulated}
    if errors:
        output["errors"] = errors
    return output
