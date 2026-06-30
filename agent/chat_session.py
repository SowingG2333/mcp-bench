"""
Chat Session Core for MCP-Bench Agent.

Provides a reusable session abstraction used by CLI, GUI, and API interfaces.
Manages LLM provider setup, MCP server connections, multi-turn conversation
history, and session persistence.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.executor import TaskExecutor
from llm.factory import LLMFactory, ModelConfig
from llm.provider import LLMProvider
from mcp_modules.server_manager_persistent import PersistentMultiServerManager
from utils.local_server_config import LocalServerConfigLoader
import config.config_loader as config_loader

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChatSessionState:
    """Serializable state of a chat session."""
    session_id: str
    created_at: float
    updated_at: float
    model_name: str
    model_config: Dict[str, Any]
    servers: List[str]
    turns: List[Dict[str, Any]]


class ChatSession:
    """Manages a single interactive conversation with the MCP agent.

    Responsibilities:
      - Build LLM provider (from env, factory, or direct endpoint args)
      - Connect to configured MCP servers and keep them alive
      - Run multi-turn dialogue using TaskExecutor
      - Persist conversation history to disk
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        server_names: Optional[List[str]] = None,
        save_dir: str = "sessions",
        filter_problematic_tools: bool = True,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model_name = model_name or "unknown"
        self.model_config = model_config
        self.llm_provider: Optional[LLMProvider] = None

        # Servers
        self.server_names = server_names or []
        self.server_configs: List[Dict[str, Any]] = []
        self.server_manager: Optional[PersistentMultiServerManager] = None
        self.filter_problematic_tools = filter_problematic_tools

        # Conversation
        self.turns: List[Turn] = []

        # Internal state
        self._initialized = False
        self._closed = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        server_names: Optional[List[str]] = None,
        save_dir: str = "sessions",
        filter_problematic_tools: bool = True,
    ) -> "ChatSession":
        """Create a session using a model registered in LLMFactory (env-based)."""
        configs = LLMFactory.get_model_configs()
        if model_name is None:
            if len(configs) == 1:
                model_name = list(configs.keys())[0]
            else:
                raise ValueError(
                    f"model_name must be specified. Available: {sorted(configs.keys())}"
                )
        if model_name not in configs:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {sorted(configs.keys())}"
            )
        return cls(
            session_id=session_id,
            model_name=model_name,
            model_config=configs[model_name],
            server_names=server_names,
            save_dir=save_dir,
            filter_problematic_tools=filter_problematic_tools,
        )

    @classmethod
    def from_direct_endpoint(
        cls,
        base_url: str,
        api_key: str,
        model_name: str,
        session_id: Optional[str] = None,
        display_name: Optional[str] = None,
        provider: str = "openai_compatible",
        server_names: Optional[List[str]] = None,
        save_dir: str = "sessions",
        filter_problematic_tools: bool = True,
    ) -> "ChatSession":
        """Create a session using a direct OpenAI-compatible endpoint."""
        model_config = ModelConfig(
            name=display_name or "custom-model",
            provider_type=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
        return cls(
            session_id=session_id,
            model_name=display_name or model_name,
            model_config=model_config,
            server_names=server_names,
            save_dir=save_dir,
            filter_problematic_tools=filter_problematic_tools,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> int:
        """Initialize LLM provider and connect to MCP servers.

        Returns the number of discovered tools.
        """
        if self._initialized:
            return len(self.server_manager.all_tools) if self.server_manager else 0

        if self.model_config is None:
            raise RuntimeError("No model_config provided. Use from_env() or from_direct_endpoint().")

        self.llm_provider = await LLMFactory.create_llm_provider(self.model_config)

        # Load server configurations
        loader = LocalServerConfigLoader()
        servers_info = loader.local_commands
        api_keys = loader.api_keys

        if not servers_info:
            raise RuntimeError("No MCP server configurations found in mcp_servers/commands.json")

        selected = self.server_names or list(servers_info.keys())
        self.server_configs = [
            cfg for name in selected
            if (cfg := self._build_server_config(name, servers_info, api_keys))
        ]

        if not self.server_configs:
            raise RuntimeError("No valid MCP server configurations. Check server names.")

        self.server_manager = PersistentMultiServerManager(
            self.server_configs,
            filter_problematic_tools=self.filter_problematic_tools,
        )
        all_tools = await self.server_manager.connect_all_servers()
        self._initialized = True
        self.server_names = [c["name"] for c in self.server_configs]
        logger.info(
            "Session %s initialized with model=%s, servers=%s, tools=%d",
            self.session_id, self.model_name, self.server_names, len(all_tools)
        )
        return len(all_tools)

    async def close(self) -> None:
        """Close MCP server connections and save final state."""
        if self._closed:
            return
        self._closed = True
        if self.server_manager:
            try:
                await self.server_manager.close_all_connections()
            except Exception as e:
                logger.warning("Error closing server connections: %s", e)
        self._save()
        logger.info("Session %s closed and saved", self.session_id)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------
    async def chat(self, message: str, timeout_seconds: int = 300) -> Dict[str, Any]:
        """Send one user message and get the agent's response.

        Returns a dict with: response, total_rounds, execution_results, tokens, etc.
        """
        if not self._initialized:
            await self.initialize()

        self.turns.append(Turn(role="user", content=message))

        # Build task prompt with conversation context
        history_text = self._format_history()
        if history_text:
            task = (
                f"Previous conversation:\n{history_text}\n\n"
                f"Current user request: {message}"
            )
        else:
            task = message

        executor = TaskExecutor(self.llm_provider, self.server_manager)
        try:
            result = await asyncio.wait_for(executor.execute(task), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            response = "⚠️ The task timed out before completion."
            self.turns.append(
                Turn(role="assistant", content=response, metadata={"error": "timeout"})
            )
            self._save()
            return {"response": response, "error": "timeout"}
        except Exception as e:
            response = f"⚠️ Execution failed: {e}"
            self.turns.append(
                Turn(role="assistant", content=response, metadata={"error": str(e)})
            )
            self._save()
            return {"response": response, "error": str(e)}

        agent_content = result.get("solution", "No solution generated.")
        execution_results = result.get("execution_results", [])
        metadata = {
            "total_rounds": result.get("total_rounds", 0),
            "tool_calls": len(execution_results),
            "planning_json_compliance": result.get("planning_json_compliance", 1.0),
            "total_tokens": result.get("total_tokens", 0),
            "prompt_tokens": result.get("total_prompt_tokens", 0),
            "completion_tokens": result.get("total_output_tokens", 0),
            "execution_results": execution_results,
        }
        self.turns.append(Turn(role="assistant", content=agent_content, metadata=metadata))
        self.updated_at = time.time()
        self._save()
        return {"response": agent_content, **metadata}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Persist session state to disk."""
        state = ChatSessionState(
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            model_name=self.model_name,
            model_config=self.model_config.config if self.model_config else {},
            servers=self.server_names,
            turns=[t.to_dict() for t in self.turns],
        )
        path = self.save_dir / f"{self.session_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save session %s: %s", self.session_id, e)

    @classmethod
    def load(cls, session_id: str, save_dir: str = "sessions") -> Optional[ChatSessionState]:
        """Load a previously saved session state (read-only metadata/history)."""
        path = Path(save_dir) / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ChatSessionState(**data)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _format_history(self, max_turns: int = 10) -> str:
        """Format recent conversation history for inclusion in the task prompt."""
        # Exclude the most recent user message (it will be appended separately)
        history_turns = [t for t in self.turns if t.role in ("user", "assistant")][-max_turns:-1]
        if not history_turns:
            return ""
        lines = []
        for turn in history_turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)

    def _build_server_config(
        self,
        server_name: str,
        servers_info: Dict[str, Any],
        api_keys: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Convert a commands.json entry to PersistentMultiServerManager config."""
        if server_name not in servers_info:
            logger.warning("Unknown server: %s", server_name)
            return None

        server_config = servers_info[server_name]
        cmd_parts = server_config.get("cmd", "").split()
        if not cmd_parts:
            logger.warning("Empty command for server: %s", server_name)
            return None

        cwd_path = server_config.get("cwd", "")
        actual_cwd = f"mcp_servers/{cwd_path[3:]}" if cwd_path.startswith("../") else cwd_path

        env = {}
        for env_var in server_config.get("env", []):
            if env_var in api_keys:
                env[env_var] = api_keys[env_var]
            elif env_var in os.environ:
                env[env_var] = os.environ[env_var]
            else:
                logger.warning("Required env var not found for %s: %s", server_name, env_var)

        config = {
            "name": server_name,
            "command": cmd_parts,
            "env": env,
            "cwd": actual_cwd,
        }

        if server_config.get("transport") == "http":
            config["transport"] = "http"
            config["port"] = server_config.get("port", config_loader.get_default_port())
            config["endpoint"] = server_config.get("endpoint", "/mcp")

        return config

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return conversation history as a list of dicts."""
        return [t.to_dict() for t in self.turns]
