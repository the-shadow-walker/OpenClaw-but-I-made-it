"""Jarvis configuration.

YAML schema mirrors BUILD_SPEC §17 exactly. Loaded from ~/.config/jarvis/config.yaml
(override via the JARVIS_CONFIG env var). Missing file is fine — defaults from the
pydantic models apply.

P0 validation: every path field is expanded (~ → $HOME), and parent directories are
created on demand (mkdir parents=True exist_ok=True). Configuration loading must
fail loudly if a path field is malformed or unwritable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "~/.config/jarvis/config.yaml"

# Published context-window sizes for chat models we know about. Keep this
# small and verified — anything not in here triggers a WARN at config load
# so drift between cfg.llm.chat_model and cfg.llm.context_window is loud.
KNOWN_MODEL_WINDOWS: dict[str, int] = {
    "qwen2.5:3b": 32768,
    "qwen3.6:35b-chain": 262144,
    # add more as deployments switch models
}


# ---------------------------------------------------------------------------
# Sub-models — one per top-level YAML section, naming matches the spec verbatim.
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5003
    workers: int = 1


class PathsConfig(BaseModel):
    workspace: Path = Path("/mnt/storage/NAS/Jarvis/jarvis/workspace")
    shared_board: Path = Path("~/.agent_bin/")

    @field_validator("workspace", "shared_board", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(str(v))))


class LLMConfig(BaseModel):
    chat_model: str = "qwen2.5:3b"
    fast_model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"
    tokenizer: Literal["qwen-native", "approximation"] = "qwen-native"
    # Bound to the chat model's published context window. Passed as
    # ``num_ctx`` on every ``ollama.chat()`` so Ollama doesn't silently
    # truncate underneath us. P6 compaction trips at trigger_pct of this.
    context_window: int = 32768

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        """Warn when chat_model/context_window drift from KNOWN_MODEL_WINDOWS.

        Operator can still override; the log makes it loud rather than silent.
        Approximation token counts already drift ±15%, so a wrong window
        compounds into wrong P6 compaction triggers — fail loud at config load.
        """
        published = KNOWN_MODEL_WINDOWS.get(self.chat_model)
        if published is None:
            logger.warning(
                "llm: unknown chat_model %r; context_window=%d may not match — "
                "verify per model and add to KNOWN_MODEL_WINDOWS",
                self.chat_model, self.context_window,
            )
        elif published != self.context_window:
            logger.info(
                "llm: chat_model %r published context window is %d, "
                "configured context_window=%d (override accepted)",
                self.chat_model, published, self.context_window,
            )


class EmbeddingProviderConfig(BaseModel):
    # P3: Ollama-only. OpenAI returns 1536-dim vectors that don't fit the
    # chunks_vec FLOAT[768] schema; rather than ship a runtime dance to
    # detect/redirect, drop the multi-provider story until P14. The user
    # has Ollama running locally on the same box; the failure mode where
    # Ollama is down but OpenAI is reachable is rare enough not to justify
    # the complexity.
    kind: Literal["ollama"]
    model: str
    dimensions: int


class EmbeddingCacheConfig(BaseModel):
    max_rows: int = 50_000


class EmbeddingsConfig(BaseModel):
    providers: list[EmbeddingProviderConfig] = Field(
        default_factory=lambda: [
            EmbeddingProviderConfig(kind="ollama", model="nomic-embed-text", dimensions=768),
        ]
    )
    cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)


class HybridSearchConfig(BaseModel):
    vector_weight: float = 0.7
    text_weight: float = 0.3
    candidate_multiplier: int = 4


class MMRConfig(BaseModel):
    lambda_: float = Field(0.7, alias="lambda")
    model_config = {"populate_by_name": True}


class DecayConfig(BaseModel):
    half_life_days: float = 30.0


class SearchConfig(BaseModel):
    hybrid: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)


class ResetConfig(BaseModel):
    daily_at: str = "04:00"           # DM convo daily reset
    group_daily_at: str = "02:00"
    idle_minutes: int = 120


class CompactionConfig(BaseModel):
    trigger_pct: float = 0.90
    keep_recent_turns: int = 6        # spec §2 invariant 5: Jarvis-side superset of contract floor (3)
    reserve_tokens_floor: int = 2_000
    auto_flush: bool = True


class ConversationConfig(BaseModel):
    reset: ResetConfig = Field(default_factory=ResetConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)


class LightSleepConfig(BaseModel):
    lookback_days: int = 7
    dedup_jaccard: float = 0.9
    min_snippet_chars: int = 20


class REMSleepConfig(BaseModel):
    lookback_days: int = 7
    cluster_min_size: int = 3


class DeepSleepWeights(BaseModel):
    relevance: float = 0.30
    frequency: float = 0.24
    query_diversity: float = 0.15
    recency: float = 0.15
    consolidation: float = 0.10
    conceptual_richness: float = 0.06

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> DeepSleepWeights:
        total = (
            self.relevance + self.frequency + self.query_diversity
            + self.recency + self.consolidation + self.conceptual_richness
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"deep_sleep.weights must sum to 1.0 (got {total:.4f})")
        return self


class DeepSleepGates(BaseModel):
    min_score: float = 0.80           # spec §2 invariant 4
    min_recall_count: int = 3
    min_unique_queries: int = 3


class DeepSleepConfig(BaseModel):
    weights: DeepSleepWeights = Field(default_factory=DeepSleepWeights)
    gates: DeepSleepGates = Field(default_factory=DeepSleepGates)
    recency_half_life_days: float = 14.0
    max_age_days: int = 30


class DreamingConfig(BaseModel):
    enabled: bool = False             # spec §21.1 — opt-in by default
    schedule: str = "0 3 * * *"       # 03:00 daily
    light_sleep: LightSleepConfig = Field(default_factory=LightSleepConfig)
    rem_sleep: REMSleepConfig = Field(default_factory=REMSleepConfig)
    deep_sleep: DeepSleepConfig = Field(default_factory=DeepSleepConfig)


class CMDOrchestrationConfig(BaseModel):
    base: str = "http://10.0.0.58:5000"
    max_concurrent: int = 2
    quick_timeout_s: int = 15
    react_max_wait_s: int = 1_800
    chain_max_wait_s: int = 7_200


class SwarmOrchestrationConfig(BaseModel):
    base: str = "http://10.0.0.58:5002"
    max_concurrent: int = 2
    dispatch_max_wait_s: int = 1_800


class OrchestrationConfig(BaseModel):
    cmd: CMDOrchestrationConfig = Field(default_factory=CMDOrchestrationConfig)
    swarm: SwarmOrchestrationConfig = Field(default_factory=SwarmOrchestrationConfig)


class MirrorConfig(BaseModel):
    enabled: bool = False             # spec §21.9 — flip after CMD env-flag patch lands
    central_context_md: Path = Path("~/.agent_bin/central_context.md")
    shared_db_path: Path = Path("~/.agent_bin/memory.db")
    poll_interval_s: float = 5.0

    @field_validator("central_context_md", "shared_db_path", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(str(v))))


class HeartbeatConfig(BaseModel):
    enabled: bool = False             # spec §21.4 — opt-in
    interval_minutes: int = 30
    checklist_path: Path = Path("workspace/HEARTBEAT.md")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class JarvisConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    dreaming: DreamingConfig = Field(default_factory=DreamingConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    mirror: MirrorConfig = Field(default_factory=MirrorConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)

    model_config = {"extra": "forbid"}   # typo guard — unknown YAML keys fail loudly


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> JarvisConfig:
    """Load the Jarvis YAML config.

    Precedence:
      1. explicit `path` argument (caller knows best)
      2. JARVIS_CONFIG env var
      3. ~/.config/jarvis/config.yaml (default)

    A missing file is fine — defaults apply. A malformed file is a hard error.
    Path fields are expanded; their parent directories are created on demand.
    """
    cfg_path: Path | None
    if path is not None:
        cfg_path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    elif "JARVIS_CONFIG" in os.environ:
        cfg_path = Path(os.path.expandvars(os.path.expanduser(os.environ["JARVIS_CONFIG"])))
    else:
        default = Path(os.path.expandvars(os.path.expanduser(DEFAULT_CONFIG_PATH)))
        cfg_path = default if default.exists() else None

    raw: dict[str, Any] = {}
    if cfg_path is not None:
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {cfg_path} must be a YAML mapping at the top level")
        raw = loaded

    cfg = JarvisConfig.model_validate(raw)

    # Path validation: only when a real config file was loaded. Defaults target the
    # production server (/mnt/storage/NAS/Jarvis/...) and won't be writable from a
    # Mac dev shell — that's expected. Production deploys always have a config file
    # under ~/.config/jarvis/config.yaml; the daemon (P5+) calls validate_paths()
    # explicitly at startup if it wants the strict check independent of file presence.
    if cfg_path is not None:
        validate_paths(cfg)

    return cfg


def validate_paths(cfg: JarvisConfig) -> None:
    """Ensure every configured directory exists or can be created. Fails loudly."""
    _ensure_dir(cfg.paths.workspace, "paths.workspace")
    _ensure_dir(cfg.paths.shared_board, "paths.shared_board")
    _ensure_dir(cfg.mirror.central_context_md.parent, "mirror.central_context_md (parent)")


def _ensure_dir(p: Path, label: str) -> None:
    """Create a directory if missing; raise with a useful label on failure."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"cannot create {label} directory at {p}: {e}") from e
    except OSError as e:
        raise OSError(f"cannot create {label} directory at {p}: {e}") from e
