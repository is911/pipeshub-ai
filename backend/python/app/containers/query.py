from dependency_injector import containers, providers

from app.config.configuration_service import ConfigurationService
from app.config.providers.etcd.etcd3_encrypted_store import Etcd3EncryptedKeyValueStore
from app.containers.container import BaseAppContainer
from app.containers.utils.utils import ContainerUtils
from app.utils.logger import create_logger


class QueryAppContainer(BaseAppContainer):
    """Dependency injection container for the query application."""

    # Override logger with service-specific name
    logger = providers.Singleton(create_logger, "query_service")
    container_utils = ContainerUtils()
    key_value_store = providers.Singleton(Etcd3EncryptedKeyValueStore, logger=logger)

    # Override config_service to use the service-specific logger
    config_service = providers.Singleton(ConfigurationService, logger=logger, key_value_store=key_value_store)

    # Override arango_client and redis_client to use the service-specific config_service
    arango_client = providers.Resource(
        BaseAppContainer._create_arango_client, config_service=config_service
    )
    redis_client = providers.Resource(
        BaseAppContainer._create_redis_client, config_service=config_service
    )
    kafka_service = providers.Singleton(lambda: None)  # Not used in query service
    arango_service = providers.Resource(
        container_utils.create_arango_service,
        logger=logger,
        arango_client=arango_client,
        config_service=config_service,
        kafka_service=kafka_service,
    )
    vector_db_service =  providers.Resource(
        container_utils.get_vector_db_service,
        config_service=config_service,
    )

    blob_store = providers.Resource(
        container_utils.create_blob_storage,
        logger=logger,
        config_service=config_service,
        arango_service=arango_service,
    )

    retrieval_service = providers.Resource(
        container_utils.create_retrieval_service,
        config_service=config_service,
        logger=logger,
        vector_db_service=vector_db_service,
        arango_service=arango_service,
        blob_store=blob_store,
    )
    # Reranker service - loads configuration from etcd (defaults to local BAAI/bge-reranker-base if not configured)
    reranker_service = providers.Resource(
        container_utils.create_reranker_service,
        config_service=config_service,
    )

    # Query-specific wiring configuration
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.routes.search",
            "app.api.routes.chatbot",
            "app.modules.retrieval.retrieval_service"
        ]
    )
