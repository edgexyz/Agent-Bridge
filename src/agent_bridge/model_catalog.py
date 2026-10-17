"""Normalized worker-advertised model catalogs, not invocation guarantees."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from agent_bridge.config import AgentConfig


class ModelInfo(BaseModel):
    id: str = Field(min_length=1)
    name: str | None = None
    is_default: bool = False
    # Worker-native levels; null means unknown, [] means explicitly no levels.
    supported_efforts: list[str] | None = None


class ModelCatalog(BaseModel):
    source: str = Field(min_length=1)
    models: list[ModelInfo]
    default_model: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> ModelCatalog:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model catalog contains duplicate IDs")
        return self


def model_discovery_capability(cfg: AgentConfig) -> dict:
    """Describe support without launching a process or querying a provider."""
    supported = cfg.protocol in {"agy", "codex"} or (
        cfg.protocol == "acp" and cfg.name in {"kimi", "dsh"}
    )
    source = {"agy": "agy models", "codex": "codex app-server model/list", "acp": "acp session/new"}
    return {
        "supported": supported,
        "source": source.get(cfg.protocol) if supported else None,
        "requires_session": supported and cfg.protocol == "acp",
    }
