import json
import os
import warnings

from typing import Any, Union, get_args, get_origin

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts

from .variables.base import BaseConfig
from .variables.default import DEFAULT_CONFIG


class Config:
    """Config class for GPT Researcher."""

    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "variables")

    def __init__(self, config_path: str | None = None):
        """Initialize the config class."""
        self.config_path = config_path
        self.llm_kwargs: dict[str, Any] = {}
        self.embedding_kwargs: dict[str, Any] = {}

        config_to_use = self.load_config(config_path)
        self._set_attributes(config_to_use)
        self._set_embedding_attributes()
        self._set_llm_attributes()
        self._handle_deprecated_attributes()
        if config_to_use["REPORT_SOURCE"] != "web":
            self._set_doc_path(config_to_use)

        # MCP support configuration
        self.mcp_servers = []  # List of MCP server configurations
        self.mcp_allowed_root_paths = []  # Allowed root paths for MCP servers

        # Read from config
        if hasattr(self, "mcp_servers"):
            self.mcp_servers = self.mcp_servers
        if hasattr(self, "mcp_allowed_root_paths"):
            self.mcp_allowed_root_paths = self.mcp_allowed_root_paths

<<<<<<< HEAD
    def _set_attributes(self, config: Dict[str, Any]) -> None:
        """
        Set instance attributes from a configuration mapping, applying environment overrides and special handling for the RETRIEVER setting.
        
        For each key/value in `config`, if an environment variable with the same name is present the value is converted using the type hint from `BaseConfig` via `convert_env_value`; the resulting value is then assigned to an attribute on `self` named after the lowercase key. After processing the mapping, the RETRIEVER value is resolved from the environment or config (defaulting to "tavily"), parsed with `parse_retrievers`, and assigned to `self.retrievers`. If parsing fails, a warning is printed and `self.retrievers` is set to ["tavily"].
        
        Parameters:
            config (Dict[str, Any]): Configuration mapping whose keys are configuration names (matching environment variable names).
        """
=======
    def _set_attributes(self, config: dict[str, Any]) -> None:
>>>>>>> newdev
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                value = self.convert_env_value(
                    key, env_value, BaseConfig.__annotations__[key]
                )
            setattr(self, key.lower(), value)

        # Handle RETRIEVER with default value
        retriever_env = os.environ.get("RETRIEVER", config.get("RETRIEVER", "tavily"))
        try:
            self.retrievers = self.parse_retrievers(retriever_env)
        except ValueError as e:
            print(f"Warning: {e!s}. Defaulting to 'tavily' retriever.")
            self.retrievers = ["tavily"]

    def _set_embedding_attributes(self) -> None:
        """
        Set the instance embedding provider and model by parsing the configured embedding string.
        
        Parses self.embedding via parse_embedding and assigns the resulting (provider, model)
        tuple to self.embedding_provider and self.embedding_model respectively.
        
        Raises:
            ValueError: If the embedding string is malformed or references an unsupported provider.
        """
        self.embedding_provider, self.embedding_model = self.parse_embedding(
            self.embedding
        )

    def _set_llm_attributes(self) -> None:
        self.fast_llm_provider, self.fast_llm_model = self.parse_llm(self.fast_llm)
        self.smart_llm_provider, self.smart_llm_model = self.parse_llm(self.smart_llm)
        self.strategic_llm_provider, self.strategic_llm_model = self.parse_llm(
            self.strategic_llm
        )
        self.reasoning_effort = self.parse_reasoning_effort(
            os.getenv("REASONING_EFFORT")
        )

    def _handle_deprecated_attributes(self) -> None:
        if os.getenv("EMBEDDING_PROVIDER") is not None:
            warnings.warn(
                "EMBEDDING_PROVIDER is deprecated and will be removed soon. Use EMBEDDING instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.embedding_provider = (
                os.environ["EMBEDDING_PROVIDER"] or self.embedding_provider
            )

            match os.environ["EMBEDDING_PROVIDER"]:
                case "ollama":
                    self.embedding_model = os.environ["OLLAMA_EMBEDDING_MODEL"]
                case "custom":
                    self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "custom")
                case "openai":
                    self.embedding_model = "text-embedding-3-large"
                case "azure_openai":
                    self.embedding_model = "text-embedding-3-large"
                case "huggingface":
                    self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                case "gigachat":
                    self.embedding_model = "Embeddings"
                case "google_genai":
                    self.embedding_model = "text-embedding-004"
                case _:
                    raise Exception("Embedding provider not found.")

        deprecation_warning = (
            "LLM_PROVIDER, FAST_LLM_MODEL and SMART_LLM_MODEL are deprecated and "
            "will be removed soon. Use FAST_LLM and SMART_LLM instead."
        )
        if os.getenv("LLM_PROVIDER") is not None:
            warnings.warn(deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.fast_llm_provider
            )
            self.smart_llm_provider = (
                os.environ["LLM_PROVIDER"] or self.smart_llm_provider
            )
        if os.getenv("FAST_LLM_MODEL") is not None:
            warnings.warn(deprecation_warning, FutureWarning, stacklevel=2)
            self.fast_llm_model = os.environ["FAST_LLM_MODEL"] or self.fast_llm_model
        if os.getenv("SMART_LLM_MODEL") is not None:
            warnings.warn(deprecation_warning, FutureWarning, stacklevel=2)
            self.smart_llm_model = os.environ["SMART_LLM_MODEL"] or self.smart_llm_model

<<<<<<< HEAD
    def _set_doc_path(self, config: Dict[str, Any]) -> None:
        """
        Set and validate the documentation path from the given configuration.
        
        Reads 'DOC_PATH' from the provided config, assigns it to self.doc_path, and—if truthy—attempts to validate (and create) the directory via validate_doc_path(). If validation fails, prints a warning and falls back to the DEFAULT_CONFIG['DOC_PATH'] value.
        
        Parameters:
            config (Dict[str, Any]): Configuration mapping expected to contain the 'DOC_PATH' key.
        """
        self.doc_path = config['DOC_PATH']
=======
    def _set_doc_path(self, config: dict[str, Any]) -> None:
        self.doc_path = config["DOC_PATH"]
>>>>>>> newdev
        if self.doc_path:
            try:
                self.validate_doc_path()
            except Exception as e:
<<<<<<< HEAD
                print(f"Warning: Error validating doc_path: {e!s}. Using default doc_path.")
                self.doc_path = DEFAULT_CONFIG['DOC_PATH']
=======
                print(
                    f"Warning: Error validating doc_path: {e!s}. Using default doc_path."
                )
                self.doc_path = DEFAULT_CONFIG["DOC_PATH"]
>>>>>>> newdev

    @classmethod
    def load_config(cls, config_path: str | None) -> dict[str, Any]:
        """Load a configuration by name."""
        if config_path is None:
            return DEFAULT_CONFIG

        # config_path = os.path.join(cls.CONFIG_DIR, config_path)
        if not os.path.exists(config_path):
            if config_path and config_path != "default":
                print(
                    f"Warning: Configuration not found at '{config_path}'. Using default configuration."
                )
                if not config_path.endswith(".json"):
                    print(f"Do you mean '{config_path}.json'?")
            return DEFAULT_CONFIG

        with open(config_path) as f:
            custom_config = json.load(f)

        # Merge with default config to ensure all keys are present
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(custom_config)
        return merged_config

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List all available configuration names."""
        configs = ["default"]
        for file in os.listdir(cls.CONFIG_DIR):
            if file.endswith(".json"):
                configs.append(file[:-5])  # Remove .json extension
        return configs

<<<<<<< HEAD
    def parse_retrievers(self, retriever_str: str) -> List[str]:
        """
        Parse a comma-separated retriever string into a validated list of retriever names.
        
        The input should be a comma-separated string of retriever identifiers (e.g. "tavily,local").
        Each name is trimmed of surrounding whitespace and must match one of the available retriever
        names returned by get_all_retriever_names().
        
        Parameters:
            retriever_str (str): Comma-separated retriever names.
        
        Returns:
            List[str]: A list of validated retriever names in the same order as provided.
        
        Raises:
            ValueError: If any retriever name is not among the available retrievers.
        """
        from ..retrievers.utils import get_all_retriever_names

        retrievers = [retriever.strip()
                      for retriever in retriever_str.split(",")]
=======
    def parse_retrievers(self, retriever_str: str) -> list[str]:
        """Parse the retriever string into a list of retrievers and validate them."""
        from ..retrievers.utils import get_all_retriever_names

        retrievers = [retriever.strip() for retriever in retriever_str.split(",")]
>>>>>>> newdev
        valid_retrievers = get_all_retriever_names() or []
        invalid_retrievers = [r for r in retrievers if r not in valid_retrievers]
        if invalid_retrievers:
            raise ValueError(
                f"Invalid retriever(s) found: {', '.join(invalid_retrievers)}. "
                f"Valid options are: {', '.join(valid_retrievers)}."
            )
        return retrievers

    @staticmethod
    def parse_llm(llm_str: str | None) -> tuple[str | None, str | None]:
        """Parse llm string into (llm_provider, llm_model)."""
        from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

        if llm_str is None:
            return None, None
        try:
            llm_provider, llm_model = llm_str.split(":", 1)
            assert llm_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {llm_provider}.\nSupported llm providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return llm_provider, llm_model
        except ValueError:
            raise ValueError(
                "Set SMART_LLM or FAST_LLM = '<llm_provider>:<llm_model>' "
                "Eg 'openai:gpt-4o-mini'"
            )

    @staticmethod
    def parse_reasoning_effort(reasoning_effort_str: str | None) -> str | None:
        """Parse reasoning effort string into (reasoning_effort)."""
        if reasoning_effort_str is None:
            return ReasoningEfforts.Medium.value
        if reasoning_effort_str not in [effort.value for effort in ReasoningEfforts]:
            raise ValueError(
                f"Invalid reasoning effort: {reasoning_effort_str}. Valid options are: {', '.join([effort.value for effort in ReasoningEfforts])}"
            )
        return reasoning_effort_str

    @staticmethod
    def parse_embedding(embedding_str: str | None) -> tuple[str | None, str | None]:
        """Parse embedding string into (embedding_provider, embedding_model)."""
        from gpt_researcher.memory.embeddings import _SUPPORTED_PROVIDERS

        if embedding_str is None:
            return None, None
        try:
            embedding_provider, embedding_model = embedding_str.split(":", 1)
            assert embedding_provider in _SUPPORTED_PROVIDERS, (
                f"Unsupported {embedding_provider}.\nSupported embedding providers are: "
                + ", ".join(_SUPPORTED_PROVIDERS)
            )
            return embedding_provider, embedding_model
        except ValueError:
            raise ValueError(
                "Set EMBEDDING = '<embedding_provider>:<embedding_model>' "
                "Eg 'openai:text-embedding-3-large'"
            )

    def validate_doc_path(self):
        """Ensure that the folder exists at the doc path"""
        os.makedirs(self.doc_path, exist_ok=True)

    @staticmethod
    def convert_env_value(key: str, env_value: str, type_hint: type) -> Any:
        """Convert environment variable to the appropriate type based on the type hint."""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            # Handle Union types (e.g., Union[str, None])
            for arg in args:
                if arg is type(None):
                    if env_value.lower() in ("none", "null", ""):
                        return None
                else:
                    try:
                        return Config.convert_env_value(key, env_value, arg)
                    except ValueError:
                        continue
            raise ValueError(f"Cannot convert {env_value} to any of {args}")

        if type_hint is bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif type_hint is int:
            return int(env_value)
        elif type_hint is float:
            return float(env_value)
        elif type_hint in (str, Any):
            return env_value
        elif origin is list or origin is list or type_hint is dict:
            return json.loads(env_value)
        else:
            raise ValueError(f"Unsupported type {type_hint} for key {key}")

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbosity level."""
        self.llm_kwargs["verbose"] = verbose

    def get_mcp_server_config(self, name: str) -> dict:
        """
<<<<<<< HEAD
        Return the configuration dict for an MCP server with the given name.
        
        If name is falsy, there are no configured servers, or no server with a matching
        exact "name" key is found, an empty dict is returned.
=======
        Get the configuration for an MCP server.

        Args:
            name (str): The name of the MCP server to get the config for.

        Returns:
            dict: The server configuration, or an empty dict if the server is not found.
>>>>>>> newdev
        """
        if not name or not self.mcp_servers:
            return {}

        for server in self.mcp_servers:
            if isinstance(server, dict) and server.get("name") == name:
                return server

<<<<<<< HEAD
        return {}
=======
        return {}
>>>>>>> newdev
