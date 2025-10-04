from pathlib import Path
import tomllib

CONFIG_PATH = Path(__file__).parent / "config.toml"

def llm_initilize(config_path: str | Path):
    """Initialize the LLM service."""
    config_dict = toml.load(Path(config_path))["llm_config"]
    
    if "code_model" not in config_dict:
        raise ValueError("Please specify a code model in the config file.")
    else:
        code_model = LLMConfig(config_dict["code_model"])

    if "embedding_model" not in config_dict:
        raise ValueError("Please specify an embedding model in the config file.")
    else:
        embedding_model = LLMConfig(config_dict["embedding_model"])

    if "reranker_model" not in config_dict:
        raise ValueError("Please specify a reranker model in the config file.")
    else:
        reranker_model = None

    return {
        "code_model": code_model,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model
    }

def learning_initialize(config_path: str | Path):
    """Initialize the learning service."""
    config_dict = toml.load(Path(config_path))["learning_config"]

    if "repo_url" not in config_dict:
        raise ValueError("Please specify a learning repo url in the config file.")
    else:
        model = LearningConfig(config_dict["model"])