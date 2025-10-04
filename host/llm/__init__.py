from pathlib import Path

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