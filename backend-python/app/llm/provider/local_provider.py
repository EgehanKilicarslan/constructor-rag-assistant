from typing import Any, Dict, List

import httpx

from ..base import LLMProvider


class LocalProvider(LLMProvider):
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url
        self.timeout = httpx.Timeout(float(timeout), connect=5.0)

    async def generate_response(
        self, query: str, context_docs: List[str], history: List[Dict[str, str]]
    ) -> str:
        system_prompt = (
            "You are a helpful and precise AI assistant. "
            "Your task is to answer the user's question based ONLY on the provided context. "
            "If the answer is not present in the context, state that you do not have enough information. "
            "Do not fabricate information or use outside knowledge unless explicitly asked."
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        context_str = "\n\n---\n\n".join(context_docs)

        user_prompt = (
            f"Please answer the question based on the following context:\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}"
        )

        messages.append({"role": "user", "content": user_prompt})

        payload = {"messages": messages, "temperature": 0.1, "max_tokens": 1024, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, json=payload)

                response.raise_for_status()

                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"].get("content", "")
                    return content
                else:
                    return "Error: Unexpected response format from Local LLM."

        except httpx.ConnectError:
            return "Error: Could not connect to Local LLM server. Is it running?"
        except httpx.TimeoutException:
            return "Error: Local LLM timed out while generating response."
        except Exception as e:
            return f"Error generating response (Local): {str(e)}"

    @property
    def provider_name(self) -> str:
        return "local"
