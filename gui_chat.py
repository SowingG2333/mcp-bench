#!/usr/bin/env python3
"""
Gradio web UI for chatting with the MCP-Bench agent.

Usage:
    python gui_chat.py
    python gui_chat.py --model gpt-4o --servers "Wikipedia,Time MCP"
    python gui_chat.py --share  # create a public gradio link
"""
import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from agent.chat_session import ChatSession
from llm.factory import LLMFactory
from utils.local_server_config import LocalServerConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global session object for the current Gradio session
_active_session: Optional[ChatSession] = None


def get_available_models() -> List[str]:
    return sorted(LLMFactory.get_model_configs().keys())


def get_available_servers() -> List[str]:
    loader = LocalServerConfigLoader()
    return sorted(loader.local_commands.keys())


async def initialize_session(
    model_choice: str,
    custom_base_url: str,
    custom_api_key: str,
    custom_model_name: str,
    server_choices: List[str],
    save_dir: str,
) -> Tuple[str, str]:
    """Initialize the chat session and return status + session ID."""
    global _active_session

    # Close previous session if any
    if _active_session:
        try:
            await _active_session.close()
        except Exception:
            pass
        _active_session = None

    # Normalize inputs
    custom_base_url = (custom_base_url or "").strip() or os.getenv("MY_MODEL_BASE_URL", "")
    custom_api_key = (custom_api_key or "").strip() or os.getenv("MY_MODEL_API_KEY", "")
    custom_model_name = (custom_model_name or "").strip() or os.getenv("MY_MODEL_NAME", "")

    try:
        if custom_base_url:
            if not custom_model_name:
                return "❌ --model-name is required for direct endpoint", ""
            _active_session = ChatSession.from_direct_endpoint(
                base_url=custom_base_url,
                api_key=custom_api_key,
                model_name=custom_model_name,
                display_name=custom_model_name,
                server_names=server_choices,
                save_dir=save_dir,
            )
        else:
            _active_session = ChatSession.from_env(
                model_name=model_choice,
                server_names=server_choices,
                save_dir=save_dir,
            )

        tool_count = await _active_session.initialize()
        return (
            f"✅ Connected to {_active_session.model_name} ({tool_count} tools).",
            f"Session ID: {_active_session.session_id}",
        )
    except Exception as e:
        logger.exception("Failed to initialize session")
        return f"❌ Initialization failed: {e}", ""


def _format_trajectory(execution_results: List[Dict[str, Any]]) -> str:
    """Render tool execution trajectory as Markdown."""
    if not execution_results:
        return "*No tool calls were recorded for this turn.*"
    lines = ["### 🛠️ Agent Execution Trace", ""]
    current_round = None
    for i, r in enumerate(execution_results, 1):
        rn = r.get("round_num", 1)
        if rn != current_round:
            lines.append(f"**Round {rn}**")
            current_round = rn
        tool = r.get("tool", "unknown")
        params = r.get("parameters", {})
        success = r.get("success", False)
        status = "✅" if success else "❌"
        lines.append(f"- {status} `{tool}`  ")
        lines.append(f"  Params: `{json.dumps(params, ensure_ascii=False)}`")
        if success:
            preview = str(r.get("result", ""))[:400]
            lines.append(f"  Result: {preview}{'...' if len(str(r.get('result', ''))) > 400 else ''}")
        else:
            lines.append(f"  Error: {r.get('error', 'unknown')}")
        lines.append("")
    return "\n".join(lines)


async def respond(
    message: str,
    history: List[Dict[str, str]],
    timeout: int,
) -> Tuple[str, List[Dict[str, str]], str, str]:
    """Handle one user message and update the chat history."""
    global _active_session

    if _active_session is None:
        return "", history, "❌ Session not initialized. Please click 'Connect' first.", ""

    result = await _active_session.chat(message, timeout_seconds=timeout)
    response = result.get("response", "No response.")
    trajectory = _format_trajectory(result.get("execution_results", []))

    # Gradio 6.0 uses the 'messages' format by default
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    stats = (
        f"Rounds: {result.get('total_rounds', 0)} | "
        f"Tool calls: {result.get('tool_calls', 0)} | "
        f"Tokens: {result.get('total_tokens', 0)}"
    )
    return "", history, stats, trajectory


async def close_session() -> str:
    global _active_session
    if _active_session:
        await _active_session.close()
        sid = _active_session.session_id
        _active_session = None
        return f"Session {sid} closed and saved."
    return "No active session."


def build_ui() -> gr.Blocks:
    available_models = get_available_models()
    available_servers = get_available_servers()
    default_servers = available_servers if len(available_servers) <= 10 else []

    with gr.Blocks(title="MCP-Bench Chat") as demo:
        gr.Markdown("# 🤖 MCP-Bench Agent Chat")
        gr.Markdown(
            "Select a model and MCP servers, then start chatting. "
            "Conversations are saved to the `sessions/` directory."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Settings")
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Registered Model",
                    interactive=True,
                )
                with gr.Accordion("Direct Endpoint (overrides model dropdown)", open=False):
                    base_url_input = gr.Textbox(
                        label="Base URL",
                        placeholder="http://localhost:8000/v1",
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="EMPTY or your key",
                    )
                    model_name_input = gr.Textbox(
                        label="Model Name",
                        placeholder="qwen2.5-7b-instruct",
                    )

                server_dropdown = gr.Dropdown(
                    choices=available_servers,
                    value=default_servers,
                    label="MCP Servers",
                    multiselect=True,
                )

                connect_btn = gr.Button("Connect", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
                session_id_text = gr.Textbox(label="Session Info", interactive=False)

                with gr.Row():
                    timeout_slider = gr.Slider(
                        minimum=30,
                        maximum=600,
                        step=30,
                        value=300,
                        label="Timeout (seconds)",
                    )
                    save_dir_input = gr.Textbox(
                        value="sessions",
                        label="Save Directory",
                    )

                close_btn = gr.Button("Close Session")
                close_status = gr.Textbox(label="Close Status", interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Plan a trip from Chengdu to Shanghai...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
                stats_text = gr.Textbox(label="Last Turn Stats", interactive=False)

                with gr.Accordion("🔍 Execution Trace", open=True):
                    trajectory_markdown = gr.Markdown(label="Execution Trace")

        # Wire up events
        connect_btn.click(
            fn=initialize_session,
            inputs=[
                model_dropdown,
                base_url_input,
                api_key_input,
                model_name_input,
                server_dropdown,
                save_dir_input,
            ],
            outputs=[status_text, session_id_text],
        )

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, timeout_slider],
            outputs=[msg_input, chatbot, stats_text, trajectory_markdown],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, timeout_slider],
            outputs=[msg_input, chatbot, stats_text, trajectory_markdown],
        )

        clear_btn.click(lambda: ([], "", ""), outputs=[chatbot, stats_text, trajectory_markdown])
        close_btn.click(fn=close_session, outputs=close_status)

        demo.load(
            lambda: (
                [],
                "Configure model/servers and click Connect",
                "",
            ),
            outputs=[chatbot, status_text, session_id_text],
        )

    return demo


async def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio web UI for MCP-Bench agent")
    parser.add_argument(
        "--model",
        default=None,
        help="Pre-select a registered model",
    )
    parser.add_argument(
        "--servers",
        default="all",
        help="Comma-separated server names, or 'all'",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    try:
        import gradio
    except ImportError:
        print("Gradio is not installed. Please run: pip install gradio")
        raise SystemExit(1)
    asyncio.run(main())
