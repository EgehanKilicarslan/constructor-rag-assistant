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

    # Mock the async property correctly
    type(mock_llm).provider_name = PropertyMock(return_value=AsyncMock(return_value="dummy")())

    # Initialize the service
    service = RagService(llm_provider=mock_llm)

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

    # 3. Check arguments
    args, kwargs = mock_llm.generate_response.call_args
    assert kwargs["query"] == "Where is Constructor University?"


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
    type(mock_llm).provider_name = PropertyMock(
        return_value=AsyncMock(return_value="BrokenProvider")()
    )

    service = RagService(llm_provider=mock_llm)
    mock_request = Mock(query="Error test", session_id="1")
    mock_context = Mock()

    # ---------------------------------------------------------
    # 2. ACT
    # ---------------------------------------------------------

    response = await service.Chat(request=mock_request, context=mock_context)

    # ---------------------------------------------------------
    # 3. ASSERT
    # ---------------------------------------------------------

    assert response.answer == "An error occurred."
