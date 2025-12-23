from unittest.mock import MagicMock, Mock, PropertyMock

import pytest
from app.services.rag_service import RagService


# --- YARDIMCI MOCK SINIFI (Streaming Yanıtlar İçin) ---
async def async_iter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_chat_success_scenario():
    """
    Senaryo: Kullanıcı soru sorar, streaming (akış) yanıt döner.
    """
    # 1. ARRANGE
    mock_llm = Mock()
    # LLM, streaming bir yanıt (Async Generator) dönmeli
    mock_llm.generate_response = MagicMock(return_value=async_iter(["Hello ", "from ", "Python!"]))
    type(mock_llm).provider_name = PropertyMock(return_value="dummy")

    mock_embedding_service = Mock()
    mock_embedding_service.search = Mock(
        return_value=[
            {
                "content": "Context info",
                "metadata": {"filename": "doc.pdf", "page": 1},
                "score": 0.9,
            }
        ]
    )

    settings = Mock()
    settings.maximum_file_size = 1024 * 1024  # 1MB

    service = RagService(settings, mock_llm, mock_embedding_service)
    mock_request = Mock(query="Test Question", session_id="123")
    mock_context = Mock()

    # 2. ACT
    # Chat bir async generator olduğu için list comprehension ile tüketiyoruz
    responses = [res async for res in service.Chat(request=mock_request, context=mock_context)]

    # 3. ASSERT
    # LLM parçaları (3 parça) + Kaynak bilgisi (1 parça) = Toplam 4 yanıt beklenir
    assert len(responses) == 4

    # Parçaların birleşimi doğru mu?
    full_answer = "".join([r.answer for r in responses])
    assert "Hello from Python!" in full_answer

    # Son mesajda kaynaklar var mı?
    last_response = responses[-1]
    assert len(last_response.source_documents) == 1
    assert last_response.source_documents[0].filename == "doc.pdf"


@pytest.mark.asyncio
async def test_upload_document_success():
    """
    Senaryo: Geçerli bir metin dosyası yüklenir.
    """
    # 1. ARRANGE
    mock_embedding_service = Mock()
    # add_documents çağrıldığında 5 chunk eklendiğini varsayalım
    mock_embedding_service.add_documents.return_value = 5

    settings = Mock()
    settings.maximum_file_size = 10_000_000  # 10MB

    service = RagService(settings, Mock(), mock_embedding_service)

    mock_request = Mock()
    mock_request.filename = "test_notes.txt"
    mock_request.file_content = b"Bu bir test icerigidir. " * 50  # Yeterli uzunlukta text

    # 2. ACT
    response = await service.UploadDocument(request=mock_request, context=Mock())

    # 3. ASSERT
    assert response.status == "success"
    assert response.chunks_count == 5
    mock_embedding_service.add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_upload_document_validation_error():
    """
    Senaryo: Desteklenmeyen dosya formatı.
    """
    settings = Mock()
    settings.maximum_file_size = 10_000_000
    service = RagService(settings, Mock(), Mock())

    mock_request = Mock()
    mock_request.filename = "virus.exe"  # Yasaklı uzantı
    mock_request.file_content = b"binary data"

    response = await service.UploadDocument(request=mock_request, context=Mock())

    assert response.status == "error"
    assert "is not supported" in response.message
