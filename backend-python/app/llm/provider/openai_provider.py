from typing import Dict, List, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
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

        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(
                cast(
                    List[ChatCompletionMessageParam],
                    [{"role": h["role"], "content": h["content"]} for h in history],
                )
            )

        context_str = "\n\n---\n\n".join(context_docs)

        user_prompt = (
            f"Please answer the question based on the following context:\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}"
        )

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            return f"Error generating response (OpenAI): {str(e)}"

    @property
    def provider_name(self) -> str:
        return "openai"
