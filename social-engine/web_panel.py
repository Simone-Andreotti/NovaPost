import contextlib
import io
import os
import traceback
import webbrowser
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

import generate as engine

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
PROMPT_FILE = ASSETS_DIR / "prompt.md"

RUN_MODES = [
    ("Full Script", "full"),
    ("Only LinkedIn Slider", "linkedin"),
    ("Only X", "x"),
    ("Only TikTok Video", "tiktok_video"),
    ("Only TikTok Voice Over (Audio)", "tiktok_audio"),
]


def _open_folder(path: Path) -> str:
    if not path.exists():
        return f"Folder not found: {path}"

    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
        return f"Opened: {path}"
    except Exception:
        webbrowser.open(path.resolve().as_uri())
        return f"Opened in browser: {path}"


def open_assets_folder() -> str:
    return _open_folder(ASSETS_DIR)


def open_output_folder() -> str:
    return _open_folder(OUTPUT_DIR)


def _load_prompt_file() -> str:
    if not PROMPT_FILE.exists():
        return ""
    return PROMPT_FILE.read_text(encoding="utf-8")


def _save_prompt_file(prompt_text: str) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(prompt_text.strip() + "\n", encoding="utf-8")


def save_prompt(prompt_text: str) -> None:
    _save_prompt_file(prompt_text)


def open_output_folder_action() -> None:
    open_output_folder()


def _nim_client() -> OpenAI:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NVIDIA_NIM_API_KEY is missing in .env")
    return OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")


def _run_pipeline(mode: str, prompt_text: str) -> str:
    if not prompt_text.strip():
        raise ValueError("Prompt is empty.")

    _save_prompt_file(prompt_text)
    engine.ensure_environment()

    logs: list[str] = []

    def log(line: str) -> None:
        logs.append(line)

    log("[setup] Prompt saved to assets/prompt.md")
    log("[1/3] Discovering assets...")
    video_path, image_paths, saved_prompt = engine.find_assets()
    log(f"Video: {video_path.name}")
    log(f"Images: {len(image_paths)}")

    log("[2/3] Generating content with NVIDIA NIM...")
    content = engine.generate_copy(saved_prompt, image_paths=image_paths, video_path=video_path)

    generated_copy_path = OUTPUT_DIR / "generated_copy.json"
    generated_copy_path.write_text(engine.json.dumps(content, indent=2), encoding="utf-8")
    log(f"Saved: {generated_copy_path.name}")

    if mode in {"full", "x"}:
        engine.write_x_outputs(content["x_post"], content["x_image_plan"])
        log("Saved: x_post.txt")

    if mode in {"full", "tiktok_video"}:
        engine.build_tiktok_video(
            video_path,
            content["tiktok_script"],
            content["tiktok_segments"],
            content["scene_cuts"],
        )
        log("Saved: tiktok_final.mp4")

    if mode in {"full", "tiktok_audio"}:
        engine.build_tiktok_voiceover_audio(video_path, content["tiktok_script"])
        log("Saved: voiceover.wav")

    if mode in {"full", "linkedin"}:
        engine.build_linkedin_carousel(image_paths, content["linkedin_slides"])
        log("Saved: linkedin_carousel.pdf")

    platform_payload = {
        "x": {
            "post": content["x_post"],
            "image_plan": content["x_image_plan"],
        },
        "linkedin": {
            "caption": " ".join([s["title"] for s in content["linkedin_slides"][:2]]).strip(),
            "tags": ["Threejs", "WebGL", "Frontend", "Performance"],
        },
        "tiktok": {
            "script": content["tiktok_script"],
            "segments": content["tiktok_segments"],
        },
    }
    platform_copy_path = OUTPUT_DIR / "platform_copy.json"
    platform_copy_path.write_text(engine.json.dumps(platform_payload, indent=2), encoding="utf-8")
    log(f"Saved: {platform_copy_path.name}")

    open_output_folder()
    log("Opened output folder.")

    return "\n".join(logs)


def run_mode(mode: str, prompt_text: str, existing_log: str | None) -> Any:
    existing_log_text = (existing_log or "").strip()
    logs = [existing_log_text] if existing_log_text else []

    def push(line: str) -> tuple[str, str, Any, Any]:
        logs.append(line)
        full = "\n".join(item for item in logs if item)
        return "Running...", full, gr.update(visible=False), gr.update(visible=True)

    yield push(f"\n=== Run mode: {mode} ===")

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            pipeline_log = _run_pipeline(mode, prompt_text)

        std_output = captured.getvalue().strip()
        if pipeline_log:
            logs.append(pipeline_log)
        if std_output:
            logs.append("[engine]\n" + std_output)

        full_log = "\n".join(item for item in logs if item)
        yield "Completed", full_log, gr.update(visible=False), gr.update(visible=True)
    except Exception as exc:
        std_output = captured.getvalue().strip()
        if std_output:
            logs.append("[engine]\n" + std_output)
        logs.append(f"ERROR: {exc}")
        logs.append(traceback.format_exc())
        full_log = "\n".join(item for item in logs if item)
        yield "Failed", full_log, gr.update(visible=False), gr.update(visible=True)


def chat_with_nim(message: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    text = (message or "").strip()
    if not text:
        return history, ""

    if history is None:
        history = []

    history.append({"role": "user", "content": text})

    try:
        client = _nim_client()
        response = client.chat.completions.create(
            model="qwen/qwen3-coder-480b-a35b-instruct",
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant in a dashboard side-chat.",
                },
                *history,
            ],
        )
        content = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": str(content).strip() or "No response."})
    except Exception as exc:
        history.append({"role": "assistant", "content": f"NIM chat error: {exc}"})

    return history, ""


CYBERPUNK_CSS = """
:root {
    --brand-black: #0e0e17;
    --brand-black-soft: #0c0c14;
    --brand-red: #f75049;
    --brand-red-soft: #ff8a84;
    --brand-cyan: #5ef6ff;
    --brand-cyan-soft: #a7fbff;
    --brand-white: #ffffff;
    --panel-fill: #10101acc;
    --line-soft: #5ef6ff55;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}

body, .gradio-container {
    background:
        radial-gradient(circle at 16% 14%, #19192b 0, transparent 40%),
        radial-gradient(circle at 84% 84%, #1a2a33 0, transparent 44%),
        linear-gradient(140deg, var(--brand-black), var(--brand-black-soft));
    color: var(--brand-white) !important;
    font-family: "Consolas", "JetBrains Mono", monospace;
}

.gradio-container {
    max-width: none !important;
    width: 100vw !important;
    height: 100vh !important;
    min-height: 100vh !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container > .main {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

#app-root {
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

#panel-shell {
    flex: 1;
    min-height: 0;
    margin: 0 !important;
    gap: 0 !important;
    border-top: 1px solid var(--line-soft);
    border-bottom: 1px solid var(--line-soft);
    background: linear-gradient(180deg, #11111d, #0f0f19);
}

#left-column,
#middle-column,
#chat-column {
    min-width: 0 !important;
    flex-basis: 0 !important;
}

.cyber-card {
    height: 100%;
    min-height: 0;
    margin: 0 !important;
    border-radius: 0 !important;
    border-right: 1px solid var(--line-soft) !important;
    background: var(--panel-fill) !important;
    padding: 12px !important;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    backdrop-filter: blur(6px);
}

#chat-column {
    border-right: none !important;
}

#left-controls, #left-log, #middle-column, #chat-column {
    flex: 1;
    min-height: 0;
}

#left-controls.hide,
#left-log.hide {
    display: none !important;
}

#left-controls, #left-log {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

#left-log > .form {
    flex: 1 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

#left-log > .form > .block {
    margin: 0 !important;
}

#run-log {
    flex: 1;
    min-height: 0;
    height: 100% !important;
}

#run-log textarea {
    height: 100% !important;
    min-height: 100% !important;
    overflow: auto !important;
}

#middle-column {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

#middle-column > .form {
    flex: 1 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    margin: 0 !important;
}

#middle-column > .form > .block,
#middle-column .auto-margin {
    margin: 0 !important;
}

#prompt-box {
    flex: 1;
    min-height: 0;
    height: 100% !important;
}

#prompt-box > label {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    min-height: 0 !important;
}

#prompt-box > label > .input-container {
    flex: 1 !important;
    min-height: 0 !important;
}

#prompt-box textarea {
    height: 100% !important;
    min-height: 100% !important;
    overflow: auto !important;
}

#chat-column {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

#chat-answer-pane, #chat-input-pane {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

#chat-box {
    flex: 1;
    min-height: 0;
}

#chat-input {
    flex: 1;
    min-height: 0;
}

#chat-input textarea {
    height: 100% !important;
    min-height: 100% !important;
    overflow: auto !important;
}

#chat-send {
    margin-top: 8px;
}

#bottom-bar {
    height: 58px;
    min-height: 58px;
    margin: 0 !important;
    padding: 8px 12px !important;
    border-top: 1px solid var(--line-soft);
    background: #0c0c14;
    align-items: center;
}

#open-output-btn {
    width: 100%;
}

button.primary {
    background: var(--brand-red) !important;
    color: var(--brand-white) !important;
    border: 1px solid var(--brand-red-soft) !important;
    font-weight: 700 !important;
}

button.secondary {
    background: transparent !important;
    color: var(--brand-cyan) !important;
    border: 1px solid var(--line-soft) !important;
}

button {
    transition: background-color 140ms ease, color 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 100ms ease;
}

button.primary:hover {
    background: #ff6f67 !important;
    border-color: #ffb2ad !important;
    box-shadow: 0 0 0 1px #ff8a8440, 0 0 18px #f7504940;
}

button.secondary:hover {
    background: #152029 !important;
    color: var(--brand-cyan-soft) !important;
    border-color: #5ef6ff99 !important;
    box-shadow: 0 0 0 1px #5ef6ff33, 0 0 16px #5ef6ff22;
}

button:active {
    transform: translateY(1px);
}

button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px #5ef6ffaa !important;
}

#left-controls .gradio-radio label,
#left-controls .gradio-radio label span {
    transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
}

#left-controls fieldset .wrap > label {
    border: 1px solid var(--line-soft);
    border-radius: 10px;
    padding: 8px 10px;
    background: #0f1320;
    display: flex;
    align-items: center;
    gap: 8px;
}

#left-controls fieldset .wrap > label:hover {
    border-color: #5ef6ffaa;
    background: #152236;
}

#left-controls fieldset .wrap > label.selected,
#left-controls fieldset .wrap > label:has(input[type="radio"]:checked) {
    border-color: var(--brand-red);
    background: linear-gradient(90deg, #2b1819, #1f1d25);
    box-shadow: inset 0 0 0 1px #ff8a8455;
}

#left-controls fieldset .wrap > label.selected span,
#left-controls fieldset .wrap > label:has(input[type="radio"]:checked) span {
    color: var(--brand-red-soft) !important;
    font-weight: 700;
}

#left-controls fieldset .wrap > label input[type="radio"] {
    accent-color: var(--brand-red);
}

textarea, input, .wrap {
    background: #0c0c14 !important;
    color: var(--brand-white) !important;
    border-color: var(--line-soft) !important;
}

label {
    color: var(--brand-cyan) !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NovaPost Control Panel", fill_width=True) as app:
        with gr.Column(elem_id="app-root"):
            with gr.Row(elem_id="panel-shell"):
                with gr.Column(scale=1, elem_id="left-column", elem_classes=["cyber-card"]):
                    with gr.Column(elem_id="left-controls", visible=True) as left_controls:
                        mode = gr.Radio(
                            label="Run Mode",
                            choices=RUN_MODES,
                            value="full",
                        )
                        run_btn = gr.Button("Run Selected Mode", variant="primary")

                    with gr.Column(elem_id="left-log", visible=False) as left_log:
                        run_status = gr.Textbox(label="Run Status", interactive=False, visible=False)
                        log_box = gr.Textbox(
                            elem_id="run-log",
                            label="Run Log",
                            lines=18,
                            interactive=False,
                        )

                with gr.Column(scale=2, elem_id="middle-column", elem_classes=["cyber-card"]):
                    prompt_box = gr.Textbox(
                        elem_id="prompt-box",
                        label="Prompt",
                        lines=20,
                        value=_load_prompt_file(),
                        placeholder="Write or paste prompt here...",
                    )
                    save_btn = gr.Button("Save Prompt", variant="secondary")

                with gr.Column(scale=1, elem_id="chat-column", elem_classes=["cyber-card"]):
                    with gr.Column(elem_id="chat-answer-pane"):
                        chat = gr.Chatbot(elem_id="chat-box", label="General Chat (NVIDIA NIM)")
                    with gr.Column(elem_id="chat-input-pane"):
                        chat_input = gr.Textbox(
                            elem_id="chat-input",
                            label="Chat Input",
                            lines=12,
                            placeholder="Ask anything...",
                        )
                        chat_send = gr.Button("Send", elem_id="chat-send", variant="primary")

            with gr.Row(elem_id="bottom-bar"):
                output_btn = gr.Button("Open Output Folder", elem_id="open-output-btn", variant="secondary")

        run_btn.click(
            fn=run_mode,
            inputs=[mode, prompt_box, log_box],
            outputs=[run_status, log_box, left_controls, left_log],
        )

        save_btn.click(fn=save_prompt, inputs=prompt_box, outputs=[])
        output_btn.click(fn=open_output_folder_action, inputs=None, outputs=[])

        chat_send.click(
            fn=chat_with_nim,
            inputs=[chat_input, chat],
            outputs=[chat, chat_input],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CYBERPUNK_CSS)
