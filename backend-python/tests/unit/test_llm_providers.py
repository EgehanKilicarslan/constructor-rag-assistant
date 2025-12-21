from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from app.llm.provider import GeminiProvider, LocalProvider, OpenAIProvider


# -----------------------------------------------------------------------------
# 1. OpenAI Provider Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_generate_response_success():
    """Test OpenAI API successful response."""
    mock_client = Mock()
    mock_create = AsyncMock()

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Hello from OpenAI"))]
    mock_create.return_value = mock_response

    mock_client.chat.completions.create = mock_create

    provider = OpenAIProvider(api_key="fake-key", model="gpt-4")
    provider.client = mock_client

    response = await provider.generate_response("Hi", [], [])

    assert response == "Hello from OpenAI"
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_api_error():
    """Test OpenAI API error handling."""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    provider = OpenAIProvider("fake", "gpt-4")
    provider.client = mock_client

    response = await provider.generate_response("Hi", [], [])
    assert "Error generating response" in response
    assert "API Error" in response


# -----------------------------------------------------------------------------
# 2. Gemini Provider Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_generate_response_success():
    """Test Gemini API successful response."""
    mock_client = Mock()
    mock_generate = AsyncMock()

    mock_response = Mock()
    mock_response.text = "Hello from Gemini"
    mock_generate.return_value = mock_response

    mock_client.models.generate_content = mock_generate

    provider = GeminiProvider("fake-key", "gemini-pro")
    provider.client = mock_client

    response = await provider.generate_response("Hi", [], [])

    assert response == "Hello from Gemini"
    mock_generate.assert_called_once()


# ------------------------------------------------------------------------------
# 3. Local Provider Tests
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_provider_success():
    """Test local Llama.cpp server successful response."""
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client_instance = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Local Llama"}}]
        }

        mock_client_instance.post.return_value = mock_response

        provider = LocalProvider(base_url="http://mock-url", timeout=10)
        response = await provider.generate_response("Hi", [], [])

        assert response == "Hello from Local Llama"

        mock_client_instance.post.assert_called_once()
        args, kwargs = mock_client_instance.post.call_args
        assert kwargs["json"]["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_local_provider_connection_error():
    """Test local provider connection error handling."""
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client_instance = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client_instance

        mock_client_instance.post.side_effect = httpx.ConnectError("Connection refused")

        provider = LocalProvider("http://bad-url", 10)
        response = await provider.generate_response("Hi", [], [])

        assert "Could not connect" in response
