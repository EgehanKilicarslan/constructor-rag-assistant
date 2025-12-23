from unittest.mock import AsyncMock, Mock

import pytest
from app.llm.provider import GeminiProvider, LocalProvider, OpenAIProvider


# --- YARDIMCI MOCK FONKSİYONU ---
async def mock_async_stream(items):
    """Async generator mocklamak için yardımcı fonksiyon"""
    for item in items:
        yield item


# -----------------------------------------------------------------------------
# 1. OpenAI Provider Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_generate_response_success():
    """Test OpenAI API successful response."""
    mock_client = Mock()

    # OpenAI yanıt formatını simüle ediyoruz
    # choices[0].delta.content yapısına uygun olmalı
    chunk1 = Mock()
    chunk1.choices = [Mock(delta=Mock(content="Hello "))]

    chunk2 = Mock()
    chunk2.choices = [Mock(delta=Mock(content="from "))]

    chunk3 = Mock()
    chunk3.choices = [Mock(delta=Mock(content="OpenAI"))]

    # Mock client'ın stream dönmesini sağla
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_async_stream([chunk1, chunk2, chunk3])
    )

    # Timeout parametresi eklendi
    provider = OpenAIProvider(api_key="fake-key", model="gpt-4", timeout=10.0)
    # Client'ı inject ediyoruz (Dependency Injection manuel yapılıyor burada)
    provider.client = mock_client

    # Cevapları topla
    chunks = []
    async for chunk in provider.generate_response("Hi", [], []):
        chunks.append(chunk)

    response = "".join(chunks)
    assert response == "Hello from OpenAI"
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_api_error():
    """Test OpenAI API error handling."""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    provider = OpenAIProvider(api_key="fake", model="gpt-4", timeout=10.0)
    provider.client = mock_client

    chunks = []
    async for chunk in provider.generate_response("Hi", [], []):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "Error generating response" in response
    assert "API Error" in response


# -----------------------------------------------------------------------------
# 2. Gemini Provider Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_generate_response_success():
    """Test Gemini API successful response."""
    mock_client = Mock()

    # Gemini yanıt formatını simüle ediyoruz (chunk.text)
    chunk1 = Mock(text="Hello ")
    chunk2 = Mock(text="from ")
    chunk3 = Mock(text="Gemini")

    # Modeller servisini mockluyoruz
    mock_client.models.generate_content_stream = AsyncMock(
        return_value=mock_async_stream([chunk1, chunk2, chunk3])
    )

    provider = GeminiProvider(api_key="fake-key", model="gemini-pro", timeout=10.0)
    provider.client = mock_client

    chunks = []
    async for chunk in provider.generate_response("Hi", [], []):
        chunks.append(chunk)

    response = "".join(chunks)
    assert response == "Hello from Gemini"
    mock_client.models.generate_content_stream.assert_called_once()


# ------------------------------------------------------------------------------
# 3. Local Provider Tests
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_provider_success():
    """
    Test local Llama.cpp server successful response.
    DÜZELTME: LocalProvider artık 'AsyncOpenAI' kullanıyor, bu yüzden
    httpx yerine OpenAI client'ını mocklamalıyız.
    """
    mock_client = Mock()

    # OpenAI formatı ile aynı yapıyı bekler
    chunk1 = Mock()
    chunk1.choices = [Mock(delta=Mock(content="Hello "))]

    chunk2 = Mock()
    chunk2.choices = [Mock(delta=Mock(content="from "))]

    chunk3 = Mock()
    chunk3.choices = [Mock(delta=Mock(content="Local Llama"))]

    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_async_stream([chunk1, chunk2, chunk3])
    )

    # LocalProvider parametreleri: base_url, model, timeout
    provider = LocalProvider(base_url="http://mock-url", model="llama-2", timeout=10.0)
    provider.client = mock_client  # Mock client'ı inject et

    chunks = []
    async for chunk in provider.generate_response("Hi", [], []):
        chunks.append(chunk)

    response = "".join(chunks)
    assert response == "Hello from Local Llama"
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_local_provider_connection_error():
    """Test local provider connection error handling."""
    mock_client = Mock()
    # Bağlantı hatasını simüle et
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Connection refused"))

    provider = LocalProvider(base_url="http://bad-url", model="llama-2", timeout=10.0)
    provider.client = mock_client

    chunks = []
    async for chunk in provider.generate_response("Hi", [], []):
        chunks.append(chunk)

    response = "".join(chunks)
    assert "Error generating response" in response
    assert "Connection refused" in response
