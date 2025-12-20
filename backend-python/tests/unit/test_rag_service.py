from unittest.mock import Mock

from app.services.rag_service import RagService


def test_chat_success_scenario():
    """
    Scenario: User asks a question.
    Expected: Service calls LLM Provider and returns the answer in gRPC format.
    """

    # ---------------------------------------------------------
    # 1. ARRANGE (Setup)
    # ---------------------------------------------------------

    # Create a fake (Mock) LLM Provider
    mock_llm = Mock()

    # Set what to return when "generate_response" is called on the LLM
    mock_llm.generate_response.return_value = "Hello from Python Test!"

    # Set what to return when provider name is queried (used for logging)
    mock_llm.provider_name = "MockGPT-4"

    # Initialize the service (passing Mock LLM via Dependency Injection)
    service = RagService(llm_provider=mock_llm)

    # Prepare a fake gRPC request
    mock_request = Mock()
    mock_request.query = "Where is Constructor University?"
    mock_request.session_id = "session_123"

    # ---------------------------------------------------------
    # 2. ACT (Action)
    # ---------------------------------------------------------

    # Call the Chat method (context parameter is not important for now, pass None)
    mock_context = Mock()  # Create a mock context
    response = service.Chat(request=mock_request, context=mock_context)

    # ---------------------------------------------------------
    # 3. ASSERT (Verification)
    # ---------------------------------------------------------

    # 1. Is the answer the text we expected?
    assert response.answer == "Hello from Python Test!"

    # 2. Was the LLM Provider's 'generate_response' method actually called?
    mock_llm.generate_response.assert_called_once()

    # 3. Are the parameters passed to LLM correct? (Was the query passed correctly?)
    # call_args holds the parameters the function was called with.
    args, kwargs = mock_llm.generate_response.call_args
    assert kwargs["query"] == "Where is Constructor University?"

    print("\n✅ [Python] Chat service test passed successfully!")


def test_chat_error_handling():
    """
    Scenario: LLM throws an error (e.g., API is down).
    Expected: Service doesn't crash and returns a generic error message.
    """

    # 1. Prepare a Mock LLM that throws an error
    mock_llm = Mock()
    mock_llm.generate_response.side_effect = Exception("API Connection Error")
    mock_llm.provider_name = "BrokenProvider"

    service = RagService(llm_provider=mock_llm)
    mock_request = Mock(query="Error test", session_id="1")

    # 2. Call the method
    mock_context = Mock()  # Create a mock context
    response = service.Chat(mock_request, context=mock_context)

    # 3. Verify
    # Our service should return "An error occurred." in the try-except block.
    assert response.answer == "An error occurred."
