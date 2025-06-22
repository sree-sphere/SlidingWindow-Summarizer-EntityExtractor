import asyncio
import os
import json
import re
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

model_name = os.environ.get("OPENAI_MODEL")
semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls

MAX_CHUNK_LENGTH = 6000  # Conservative buffer under 8K token limit

# Entity types to extract
ENTITY_TYPES = ["Person", "Organization", "Location", "Date", "School", "Vehicle"]

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


async def retry_extraction(chunk_text: str, entity_type: str, system_prompt: str) -> Dict[str, Any]:
    """
    A simplified fallback extraction: ask the model to return names only, separated by commas.
    Relationship and type will be filled as 'Manual extraction fallback' for clarity.
    """
    try:
        simplified_prompt = (
            f"Extract ONLY the names of {entity_type} entities from the text, separated by commas. "
            "Do NOT include any JSON formatting, markdown or code fences—just a plain comma-separated list of names."
        )
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": simplified_prompt + "\n\n" + chunk_text}
            ]
        )
        content = response.choices[0].message.content.strip()
        # The response should be a comma-separated list. Split and strip.
        names = [name.strip() for name in content.split(",") if name.strip()]
        # Filter out obviously malformed fragments (e.g., containing braces) if any
        cleaned = []
        for name in names:
            # Skip fragments that look like JSON or code fences
            if re.search(r"[{}\[\]```]", name):
                continue
            cleaned.append(name)
        return {
            "entities": [
                {
                    "name": name,
                    "type": entity_type,
                    "relationship_to_main": "Manual extraction fallback"
                }
                for name in cleaned
            ]
        }
    except Exception as e:
        # If even retry fails, return empty with error
        return {"error": f"Retry failed: {str(e)}", "entities": []}


async def extract_entities_from_chunk_by_type(
    chunk_text: str,
    system_prompt: str,
    entity_type: str,
    temperature=0.3
) -> Dict[str, Any]:
    """
    Extract only one entity type from a single text chunk.
    Returns parsed JSON or error object. Uses a strict prompt and strips markdown fences.
    """
    async with semaphore:
        try:
            # Instruction prompt: emphasize strict JSON only, no markdown or code fences.
            instruction_prompt = (
                f"Extract only {entity_type} entities from the text.\n"
                "Return STRICTLY a JSON object matching this schema, and do NOT include any additional text, markdown, or code fences:\n"
                "{\n"
                "  \"entities\": [\n"
                "    {\n"
                "      \"name\": \"Entity Name\",\n"
                "      \"type\": \"Entity Type\",\n"
                "      \"relationship_to_main\": \"Relationship Description\"\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "Ensure the JSON is well-formed and exactly matches the schema without wrappers or markdown fences."
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction_prompt + "\n\n" + chunk_text}
                ],
                temperature=temperature
            )

            raw_content = response.choices[0].message.content.strip()

            # Strip common markdown/code fences if present
            content = raw_content
            if content.startswith("```"):
                # Try to extract the JSON block inside backticks
                # Split by ``` and find the part that looks like JSON
                parts = content.split("```")
                json_part = None
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") and part.endswith("}"):
                        json_part = part
                        break
                if json_part:
                    content = json_part
                else:
                    # As fallback, remove leading/trailing fences
                    content = re.sub(r"^```(?:json)?", "", content)
                    content = re.sub(r"```$", "", content).strip()

            # Attempt to parse JSON
            try:
                parsed = json.loads(content)
                # Validate schema
                validate(instance=parsed, schema=ENTITY_SCHEMA)
                return parsed
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[DEBUG] Failed to parse/validate JSON for entity_type={entity_type}. Error: {e}. Raw content:\n{raw_content}")
                # Retry fallback extraction
                return await retry_extraction(chunk_text, entity_type, system_prompt)

        except Exception as e:
            return {"error": str(e), "entities": []}


def split_into_chunks(texts: List[str], max_length: int) -> List[List[str]]:
    """
    Splits a list of text paragraphs into groups where the total length of joined text per group
    does not exceed max_length. Returns list of groups (each group is a list of paragraphs).
    """
    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_length = 0

    for text in texts:
        length = len(text)
        if current_chunk and (current_length + length > max_length):
            chunks.append(current_chunk)
            current_chunk = [text]
            current_length = length
        else:
            current_chunk.append(text)
            current_length += length

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def hierarchical_extract_entities(
    paragraphs: List[str],
    system_prompt: str,
    input_prompt: str = "",
    temperature=0.3
) -> Dict[str, Any]:
    """
    Top-level function: splits paragraphs into chunks, calls extraction per entity type concurrently,
    aggregates, deduplicates by name, and returns {"entities": [...]}.
    """
    # Split paragraphs into chunks of text
    chunk_groups = split_into_chunks(paragraphs, MAX_CHUNK_LENGTH)

    # Prepare tasks
    all_tasks = []
    for group in chunk_groups:
        chunk_text = "\n\n".join(group)
        for entity_type in ENTITY_TYPES:
            task = extract_entities_from_chunk_by_type(
                chunk_text=chunk_text,
                system_prompt=system_prompt,
                entity_type=entity_type,
                temperature=temperature
            )
            all_tasks.append(task)

    # Run all tasks concurrently
    results = await asyncio.gather(*all_tasks)

    # Aggregate all entities and collect errors
    all_entities: List[Dict[str, Any]] = []
    error_messages: List[str] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        if "error" in result:
            error_messages.append(result["error"])
        else:
            # Expect result to have key "entities" with a list
            ents = result.get("entities", [])
            if isinstance(ents, list):
                all_entities.extend(ents)

    # Deduplicate entities by exact name match (case-sensitive). 
    unique_entities: Dict[str, Dict[str, Any]] = {}
    for entity in all_entities:
        name = entity.get("name")
        if name:
            if name not in unique_entities:
                unique_entities[name] = entity
            # If desired, one could merge/choose the one with more detailed relationship, but here we keep first.

    output = {"entities": list(unique_entities.values())}
    if error_messages:
        output["errors"] = error_messages

    return output
