from dependency_injector import containers, providers

from .config import settings
from .llm.factory import get_llm_provider
from .services.rag_service import RagService


class Container(containers.DeclarativeContainer):
    config = providers.Object(settings)

    llm_client = providers.Factory(get_llm_provider, settings=config)

    rag_service = providers.Factory(RagService, llm_provider=llm_client)
