#!/usr/bin/env python3
"""
Interactive chat with MCP-Bench agent.

Connects to configured MCP servers and lets you have a multi-round conversation
with the TaskExecutor agent without running the full benchmark.

Usage with registered model:
    python chat_with_agent.py --model gpt-4o

Usage with direct OpenAI-compatible endpoint:
    python chat_with_agent.py --base-url http://localhost:8000/v1 --api-key EMPTY --model-name qwen2.5-7b-instruct

Or set environment variables in .env or shell:
    MY_MODEL_BASE_URL=http://localhost:8000/v1
    MY_MODEL_API_KEY=EMPTY
    MY_MODEL_NAME=qwen2.5-7b-instruct
"""
import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agent.executor import TaskExecutor
from mcp_modules.server_manager_persistent import PersistentMultiServerManager
from llm.provider import LLMProvider
from llm.factory import LLMFactory, ModelConfig
from utils.local_server_config import LocalServerConfigLoader
import config.config_loader as config_loader


def load_server_config(server_name: str, servers_info: Dict[str, Any], api_keys: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Convert commands.json entry to PersistentMultiServerManager config."""
    if server_name not in servers_info:
        return None

    server_config = servers_info[server_name]
    cmd_parts = server_config.get('cmd', '').split()
    if not cmd_parts:
        logger.warning(f"Empty command for server: {server_name}")
        return None

    cwd_path = server_config.get('cwd', '')
    if cwd_path.startswith('../'):
        actual_cwd = f"mcp_servers/{cwd_path[3:]}"
    else:
        actual_cwd = cwd_path

    env = {}
    for env_var in server_config.get('env', []):
        if env_var in api_keys:
            env[env_var] = api_keys[env_var]
        elif env_var in os.environ:
            env[env_var] = os.environ[env_var]

    config = {
        'name': server_name,
        'command': cmd_parts,
        'env': env,
        'cwd': actual_cwd
    }

    if server_config.get('transport') == 'http':
        config['transport'] = 'http'
        config['port'] = server_config.get('port', config_loader.get_default_port())
        config['endpoint'] = server_config.get('endpoint', '/mcp')

    return config


async def chat_loop(
    llm_provider: LLMProvider,
    server_manager: PersistentMultiServerManager,
    timeout_seconds: int
) -> None:
    """Run interactive conversation loop with the agent."""
    print("\n🤖 MCP Agent is ready. Type your task or 'quit' / 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in {'quit', 'exit', 'q'}:
            print("Exiting...")
            break

        executor = TaskExecutor(llm_provider, server_manager)
        try:
            result = await asyncio.wait_for(
                executor.execute(user_input),
                timeout=timeout_seconds
            )
            print("\n🧠 Agent:\n" + result.get('solution', 'No solution generated.'))
            print(f"\n[Used {result.get('total_rounds', 0)} round(s), "
                  f"{len(result.get('execution_results', []))} tool call(s)]\n")
        except asyncio.TimeoutError:
            print("\n⚠️ Task timed out.\n")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            print(f"\n⚠️ Execution failed: {e}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interactively with MCP-Bench agent")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name registered in llm/factory.py (use --list-models to see options)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Direct OpenAI-compatible API base URL. Fallback: MY_MODEL_BASE_URL env var."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the direct model endpoint. Fallback: MY_MODEL_API_KEY env var."
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Actual model name to send to the API. Fallback: MY_MODEL_NAME env var."
    )
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=["openai_compatible", "openrouter", "azure"],
        help="Provider type for --base-url (default: openai_compatible)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--servers",
        default="all",
        help="Comma-separated server names to connect, or 'all' (default: all)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Allow MY_MODEL_* env vars to fill in direct endpoint arguments
    if not args.base_url:
        args.base_url = os.getenv("MY_MODEL_BASE_URL")
    if not args.api_key:
        args.api_key = os.getenv("MY_MODEL_API_KEY")
    if not args.model_name:
        args.model_name = os.getenv("MY_MODEL_NAME")

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load model configs
    model_configs = LLMFactory.get_model_configs()

    if args.list_models:
        print("Available models (from environment / factory config):")
        for name in sorted(model_configs.keys()):
            cfg = model_configs[name]
            print(f"  - {name} ({cfg.provider_type}: {cfg.config.get('model_name', cfg.config.get('deployment_name', 'unknown'))})")
        if not model_configs:
            print("  (none - set API keys or use --base-url / MY_MODEL_* env vars)")
        return

    # Direct endpoint mode: bypass factory config
    if args.base_url:
        if not args.model_name:
            print("--model-name is required when using --base-url")
            sys.exit(1)
        model_config = ModelConfig(
            name=args.model or "custom-model",
            provider_type=args.provider,
            api_key=args.api_key or "",
            base_url=args.base_url,
            model_name=args.model_name
        )
        print(f"Using direct endpoint: {args.base_url} ({args.model_name})")
    else:
        if not model_configs:
            print("No models configured. Set API keys, use --base-url, or set MY_MODEL_* env vars.")
            sys.exit(1)

        if args.model is None:
            if len(model_configs) == 1:
                args.model = list(model_configs.keys())[0]
            else:
                print("Multiple models available, please specify one with --model:")
                for name in sorted(model_configs.keys()):
                    print(f"  - {name}")
                sys.exit(1)

        if args.model not in model_configs:
            print(f"Unknown model: {args.model}")
            print("Available models:", ", ".join(sorted(model_configs.keys())))
            sys.exit(1)

        model_config = model_configs[args.model]
        print(f"Using model: {args.model}")

    # Create LLM provider
    llm_provider = await LLMFactory.create_llm_provider(model_config)

    # Load server configurations
    loader = LocalServerConfigLoader()
    servers_info = loader.local_commands
    api_keys = loader.api_keys

    if not servers_info:
        print("No MCP server configurations found. Run python utils/collect_mcp_info.py first?")
        sys.exit(1)

    # Determine which servers to connect
    if args.servers.lower() == 'all':
        selected_servers = list(servers_info.keys())
    else:
        selected_servers = [s.strip() for s in args.servers.split(',')]

    server_configs = []
    for name in selected_servers:
        cfg = load_server_config(name, servers_info, api_keys)
        if cfg:
            server_configs.append(cfg)
        else:
            logger.warning(f"Skipping unknown/misconfigured server: {name}")

    if not server_configs:
        print("No valid server configurations. Check --servers argument.")
        sys.exit(1)

    print(f"Connecting to {len(server_configs)} server(s): {', '.join(c['name'] for c in server_configs)}...")

    # Connect to servers
    server_manager = PersistentMultiServerManager(server_configs)
    try:
        all_tools = await server_manager.connect_all_servers()
        print(f"Connected. Discovered {len(all_tools)} tool(s) total.\n")

        await chat_loop(llm_provider, server_manager, args.timeout)
    finally:
        print("\nClosing server connections...")
        try:
            await server_manager.close_all_connections()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
