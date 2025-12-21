from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest
from app.services.rag_service import RagService


@pytest.mark.asyncio
async def test_chat_success_scenario():
    """
    Scenario: User asks a question.
    Expected: Service calls LLM Provider and returns the answer in gRPC format.
    """

    # ---------------------------------------------------------
    # 1. ARRANGE (Setup)
    # ---------------------------------------------------------

    # Create a fake (Mock) LLM Provider
    mock_llm = Mock()
    mock_llm.generate_response = AsyncMock(return_value="Hello from Python Test!")
    type(mock_llm).provider_name = PropertyMock(return_value="dummy")

    # Create a fake Embedding Service
    mock_embedding_service = Mock()
    mock_embedding_service.search = Mock(
        return_value=[
            {
                "content": "Constructor University is located in Bremen, Germany.",
                "metadata": {"filename": "test.pdf", "page": 1},
                "score": 0.95,
            }
        ]
    )

    # Initialize the service
    service = RagService(llm_provider=mock_llm, embedding_service=mock_embedding_service)

    # Prepare a fake gRPC request
    mock_request = Mock()
    mock_request.query = "Where is Constructor University?"
    mock_request.session_id = "session_123"

    # Create a mock context
    mock_context = Mock()

    # ---------------------------------------------------------
    # 2. ACT (Action)
    # ---------------------------------------------------------

    # Call the Chat method asynchronously
    response = await service.Chat(request=mock_request, context=mock_context)

    # ---------------------------------------------------------
    # 3. ASSERT (Verification)
    # ---------------------------------------------------------

    # 1. Is the answer the text we expected?
    assert response.answer == "Hello from Python Test!"

    # 2. Was the LLM Provider's 'generate_response' method actually called?
    mock_llm.generate_response.assert_called_once()

    # 3. Was the embedding service search called?
    mock_embedding_service.search.assert_called_once_with(
        "Where is Constructor University?", limit=3
    )

    # 4. Check that source documents were populated
    assert len(response.source_documents) == 1
    assert response.source_documents[0].filename == "test.pdf"


@pytest.mark.asyncio
async def test_chat_error_handling():
    """
    Scenario: LLM throws an error (e.g., API is down).
    Expected: Service doesn't crash and returns a generic error message.
    """

    # ---------------------------------------------------------
    # 1. ARRANGE
    # ---------------------------------------------------------
    mock_llm = Mock()
    # Side effect exception
    mock_llm.generate_response = AsyncMock(side_effect=Exception("API Connection Error"))
    type(mock_llm).provider_name = PropertyMock(return_value="BrokenProvider")

    # Create a fake Embedding Service that returns valid search results
    mock_embedding_service = Mock()
    mock_embedding_service.search = Mock(return_value=[])

    service = RagService(llm_provider=mock_llm, embedding_service=mock_embedding_service)
    mock_request = Mock(query="Error test", session_id="1")
    mock_context = Mock()

    # ---------------------------------------------------------
    # 2. ACT
    # ---------------------------------------------------------

    response = await service.Chat(request=mock_request, context=mock_context)

    # ---------------------------------------------------------
    # 3. ASSERT
    # ---------------------------------------------------------

    assert response.answer == "Sorry, an error occurred while generating the response."
    assert response.processing_time_ms == 0.0
