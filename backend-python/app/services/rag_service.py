from typing import List

import grpc
from pb import rag_service_pb2, rag_service_pb2_grpc

from ..llm.base import LLMProvider


class RagService(rag_service_pb2_grpc.RagServiceServicer):
    def __init__(self, llm_provider: LLMProvider):
        self.llm: LLMProvider = llm_provider

    async def Chat(
        self, request: rag_service_pb2.ChatRequest, context: grpc.aio.ServicerContext
    ) -> rag_service_pb2.ChatResponse:
        print(f"[RagService] Request received. LLM: {self.llm.provider_name}")

        # Mock documents (We will inject VectorDB Service here in the future)
        mock_docs: List[rag_service_pb2.Source] = [
            rag_service_pb2.Source(
                filename="Document A",
                page_number=1,
                snippet="This is a snippet from Document A.",
                score=0.95,
            ),
            rag_service_pb2.Source(
                filename="Document B",
                page_number=2,
                snippet="This is a snippet from Document B.",
                score=0.90,
            ),
        ]

        try:
            answer: str = await self.llm.generate_response(
                query=request.query, context_docs=[doc.snippet for doc in mock_docs], history=[]
            )
        except Exception as e:
            print(f"Error: {e}")
            answer = "An error occurred."

        return rag_service_pb2.ChatResponse(
            answer=answer, source_documents=mock_docs, processing_time_ms=10.0
        )
