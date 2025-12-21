from typing import Dict, List

from google.genai import Client
from google.genai.types import GenerateContentConfig

from ..base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Client(api_key=api_key).aio
        self.model = model

    async def generate_response(
        self, query: str, context_docs: List[str], history: List[Dict[str, str]]
    ) -> str:
        system_prompt = (
            "You are a helpful and precise AI assistant. "
            "Your task is to answer the user's question based ONLY on the provided context. "
            "If the answer is not present in the context, state that you do not have enough information. "
            "Do not fabricate information or use outside knowledge unless explicitly asked."
        )

        messages = []
        if history:
            messages.extend(history)

        context_str = "\n\n---\n\n".join(context_docs)

        user_prompt = (
            f"\nPlease answer the question based on the following context:\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}"
        )

        messages.append(user_prompt)

        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    temperature=0.1,
                ),
            )

            if response.text is None:
                raise ValueError("Response text is None")

            return response.text
        except Exception as e:
            return f"Error generating response (Gemini): {str(e)}"

    @property
    def provider_name(self) -> str:
        return "gemini"
