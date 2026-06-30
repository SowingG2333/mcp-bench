#!/usr/bin/env python3
"""
Interactive CLI chat with MCP-Bench agent.

Usage with registered model:
    python chat_with_agent.py --model gpt-4o

Usage with direct OpenAI-compatible endpoint:
    python chat_with_agent.py --base-url http://localhost:8000/v1 --api-key EMPTY --model-name qwen2.5-7b-instruct

Or set environment variables in .env or shell:
    MY_MODEL_BASE_URL=http://localhost:8000/v1
    MY_MODEL_API_KEY=EMPTY
    MY_MODEL_NAME=qwen2.5-7b-instruct
"""
import argparse
import asyncio
import logging
import os
import sys
from typing import List

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.chat_session import ChatSession
from llm.factory import LLMFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_server_list(value: str) -> List[str]:
    if not value or value.lower() == "all":
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interactively with MCP-Bench agent")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name registered in llm/factory.py (use --list-models to see options)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Direct OpenAI-compatible API base URL. Fallback: MY_MODEL_BASE_URL env var.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the direct model endpoint. Fallback: MY_MODEL_API_KEY env var.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Actual model name to send to the API. Fallback: MY_MODEL_NAME env var.",
    )
    parser.add_argument(
        "--provider",
        default="openai_compatible",
        choices=["openai_compatible", "openrouter", "azure"],
        help="Provider type for --base-url (default: openai_compatible)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--servers",
        default="all",
        help="Comma-separated server names to connect, or 'all' (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--save-dir",
        default="sessions",
        help="Directory to save conversation sessions (default: sessions)",
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

    # List models
    model_configs = LLMFactory.get_model_configs()
    if args.list_models:
        print("Available models (from environment / factory config):")
        for name in sorted(model_configs.keys()):
            cfg = model_configs[name]
            print(
                f"  - {name} ({cfg.provider_type}: "
                f"{cfg.config.get('model_name', cfg.config.get('deployment_name', 'unknown'))})"
            )
        if not model_configs:
            print("  (none - set API keys or use --base-url / MY_MODEL_* env vars)")
        return

    # Build session
    server_names = parse_server_list(args.servers)

    if args.base_url:
        if not args.model_name:
            print("--model-name is required when using --base-url")
            sys.exit(1)
        session = ChatSession.from_direct_endpoint(
            base_url=args.base_url,
            api_key=args.api_key or "",
            model_name=args.model_name,
            display_name=args.model or args.model_name,
            provider=args.provider,
            server_names=server_names,
            save_dir=args.save_dir,
        )
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

        try:
            session = ChatSession.from_env(
                model_name=args.model,
                server_names=server_names,
                save_dir=args.save_dir,
            )
        except ValueError as e:
            print(str(e))
            sys.exit(1)

    # Initialize (connect servers)
    try:
        tool_count = await session.initialize()
        print(f"\n🤖 MCP Agent is ready ({session.model_name}, {tool_count} tools).")
        print(f"   Session ID: {session.session_id}")
        print("   Type your task or 'quit' / 'exit' to stop.\n")
    except Exception as e:
        logger.error("Failed to initialize session: %s", e)
        await session.close()
        sys.exit(1)

    # Chat loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Exiting...")
                break

            result = await session.chat(user_input, timeout_seconds=args.timeout)
            print("\n🧠 Agent:\n" + result.get("response", "No response."))
            print(
                f"\n[Used {result.get('total_rounds', 0)} round(s), "
                f"{result.get('tool_calls', 0)} tool call(s)]\n"
            )
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
