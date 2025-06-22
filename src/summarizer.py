import asyncio
import os
from openai import AsyncOpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
    )

model_name = os.environ.get("OPENAI_MODEL")
semaphore = asyncio.Semaphore(10)  # Limit concurrency to 5


async def summarize_paragraph(paragraph: str, system_prompt: str, instruction_prompt: str, temperature=0.3) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{instruction_prompt}\n\n{paragraph}"}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error summarizing paragraph: {str(e)}]"


MAX_CHUNK_LENGTH = 4000  # Rough estimate of max combined token limit (you can tune this)


def split_into_chunks(texts: List[str], max_length: int) -> List[List[str]]:
    """Split a list of texts into chunks that won't exceed max_length when combined."""
    chunks = []
    current_chunk = []
    current_length = 0

    for text in texts:
        length = len(text)
        if current_length + length > max_length:
            chunks.append(current_chunk)
            current_chunk = [text]
            current_length = length
        else:
            current_chunk.append(text)
            current_length += length

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def hierarchical_summarize(
    paragraphs: List[str], system_prompt: str, instruction_prompt: str, temperature=0.3
) -> str:
    # First level summaries
    layer = 1
    current_texts = paragraphs
    while True:
        print(f"[Layer {layer}] {len(current_texts)} chunks to summarize")
        # Break into chunks that won't overflow context
        chunk_groups = split_into_chunks(current_texts, MAX_CHUNK_LENGTH)

        # Summarize each group
        summarized_chunks = []
        for group in chunk_groups:
            combined = "\n\n".join(group)
            summary = await summarize_paragraph(
                combined,
                system_prompt,
                f"{instruction_prompt}\n\nSummarize this chunk:",
                temperature,
            )
            summarized_chunks.append(summary)

        # If only one summary, we're done
        if len(summarized_chunks) == 1:
            return summarized_chunks[0]

        # Prepare for next layer
        current_texts = summarized_chunks
        layer += 1