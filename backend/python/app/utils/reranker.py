from typing import Tuple

from app.config.configuration_service import ConfigurationService
from app.config.constants.service import config_node_constants
from app.modules.reranker.reranker import (
    BaseRerankerService,
    LocalRerankerService,
    get_reranker_service,
)
from app.utils.logger import create_logger

logger = create_logger("reranker_utils")


async def get_reranker(
    config_service: ConfigurationService,
    reranker_configs: list = None,
) -> Tuple[BaseRerankerService, dict]:
    """
    Get the configured reranker service from etcd configuration.

    Args:
        config_service: Configuration service for fetching from etcd
        reranker_configs: Optional list of reranker configs (if already fetched)

    Returns:
        Tuple of (reranker_service, config_dict)
    """
    if not reranker_configs:
        try:
            ai_models = await config_service.get_config(
                config_node_constants.AI_MODELS.value,
                use_cache=False
            )
            reranker_configs = ai_models.get("reranker", [])
        except Exception as e:
            logger.warning(f"Failed to fetch reranker config from etcd: {e}. Using default local reranker.")
            reranker_configs = []

    # If no reranker configured, use default local reranker
    if not reranker_configs:
        logger.info("No reranker configured, using default local reranker (BAAI/bge-reranker-base)")
        default_config = {
            "provider": "local",
            "configuration": {
                "model": "BAAI/bge-reranker-base"
            },
            "isDefault": True
        }
        return LocalRerankerService(model_name="BAAI/bge-reranker-base"), default_config

    # Find default reranker config
    for config in reranker_configs:
        if config.get("isDefault", False):
            logger.info(f"Using default reranker: provider={config['provider']}")
            reranker = get_reranker_service(config["provider"], config)
            return reranker, config

    # If no default, use first available
    if reranker_configs:
        config = reranker_configs[0]
        logger.info(f"Using first available reranker: provider={config['provider']}")
        reranker = get_reranker_service(config["provider"], config)
        return reranker, config

    # Fallback to local reranker
    logger.info("Falling back to default local reranker")
    default_config = {
        "provider": "local",
        "configuration": {
            "model": "BAAI/bge-reranker-base"
        },
        "isDefault": True
    }
    return LocalRerankerService(model_name="BAAI/bge-reranker-base"), default_config
