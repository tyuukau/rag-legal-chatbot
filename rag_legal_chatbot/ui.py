import os
import shutil
import json
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar

from llama_index.core.schema import MetadataMode
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from .pipeline import LocalRAGPipeline
from .logger import Logger


_JS_LIGHT_THEME = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

_CSS = """
.btn {
    background-color: #64748B;
    color: #FFFFFF;
    }

.stop_btn {
    background-color: #ff7373;
    color: #FFFFFF;
    }
"""


@dataclass
class _DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    SET_MODEL_MESSAGE: str = "You need to choose LLM model 🤖 first!"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    CONFIRM_PULL_MODEL_STATUS: str = "Confirm Pull Model!"
    PULL_MODEL_SCUCCESS_STATUS: str = "Pulling model 🤖 completed!"
    PULL_MODEL_FAIL_STATUS: str = "Pulling model 🤖 failed!"
    MODEL_NOT_EXIST_STATUS: str = "Model doesn't exist!"
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = (
        "Processing documents 📄 completed!"
    )
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"


class _LLMResponse:
    def __init__(self) -> None:
        # Pass
        pass

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                _DefaultElement.DEFAULT_MESSAGE,
                [[None, message[: i + 1]]],
                _DefaultElement.DEFAULT_STATUS,
            )

    def yield_welcome_string(self):
        yield from self._yield_string(_DefaultElement.HELLO_MESSAGE)

    def yield_set_model_string(self):
        yield from self._yield_string(_DefaultElement.SET_MODEL_MESSAGE)

    def yield_empty_message_string(self):
        yield from self._yield_string(_DefaultElement.EMPTY_MESSAGE)

    def yield_stream_response(
        self,
        message: str,
        history: list[list[str]],
        response: StreamingAgentChatResponse,
    ):
        answer = []
        for text in response.response_gen:
            answer.append(text)
            yield (
                _DefaultElement.DEFAULT_MESSAGE,
                history + [[message, "".join(answer)]],
                _DefaultElement.ANSWERING_STATUS,
            )
        yield (
            _DefaultElement.DEFAULT_MESSAGE,
            history + [[message, "".join(answer)]],
            _DefaultElement.COMPLETED_STATUS,
        )


class LocalChatbotUI:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        host: str = "host.docker.internal",
        data_dir: str = "data/data",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
    ):
        self.pipeline = pipeline
        self.logger = logger
        self.host = host
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = _LLMResponse()
        self._sources: list[str] = []

    def _get_sources(self):
        return self._sources

    def _get_respone(
        self,
        chat_mode: str,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        if self.pipeline.get_model_name() in [None, ""]:
            for m in self._llm_response.yield_set_model_string():
                yield *m, ""
            self._sources = []
        elif message["text"] in [None, ""]:
            for m in self._llm_response.yield_empty_message_string():
                yield *m, ""
            self._sources = []
        else:
            console = sys.stdout
            sys.stdout = self.logger
            response = self.pipeline.query(
                chat_mode, message["text"], chatbot
            )
            for m in self._llm_response.yield_stream_response(
                message["text"], chatbot, response
            ):
                yield *m, "\n\n".join(
                    [
                        n.node.get_content(
                            metadata_mode=MetadataMode.LLM
                        ).strip()
                        for n in response.source_nodes
                    ]
                )
            sys.stdout = console
            self._sources = [
                n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                for n in response.source_nodes
            ]

    async def _aget_respone(
        self,
        chat_mode: str,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        if self.pipeline.get_model_name() in [None, ""]:
            for m in self._llm_response.yield_set_model_string():
                yield m
            self._sources = []
        elif message["text"] in [None, ""]:
            for m in self._llm_response.yield_empty_message_string():
                yield m
            self._sources = []
        else:
            console = sys.stdout
            sys.stdout = self.logger
            response = await self.pipeline.aquery(
                chat_mode, message["text"], chatbot
            )
            for m in self._llm_response.yield_stream_response(
                message["text"], chatbot, response
            ):
                yield m
            sys.stdout = console
            self._sources = [
                n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                for n in response.source_nodes
            ]

    def _get_confirm_pull_model(self, model: str):
        if (model in ["gpt-4o-mini", "gpt-4o"]) or (
            self.pipeline.check_exist(model)
        ):
            self._change_model(model)
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                _DefaultElement.DEFAULT_STATUS,
            )
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            _DefaultElement.CONFIRM_PULL_MODEL_STATUS,
        )

    def _pull_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        if (model not in ["gpt-4o-mini", "gpt-4o"]) and not (
            self.pipeline.check_exist(model)
        ):
            response = self.pipeline.pull_model(model)
            if response.status_code == 200:
                gr.Info(f"Pulling {model}!")
                for data in response.iter_lines(chunk_size=1):
                    data = json.loads(data)
                    if "completed" in data.keys() and "total" in data.keys():
                        progress(
                            data["completed"] / data["total"],
                            desc="Downloading",
                        )
                    else:
                        progress(0.0)
            else:
                gr.Warning(f"Model {model} doesn't exist!")
                return (
                    _DefaultElement.DEFAULT_MESSAGE,
                    _DefaultElement.DEFAULT_HISTORY,
                    _DefaultElement.PULL_MODEL_FAIL_STATUS,
                    _DefaultElement.DEFAULT_MODEL,
                )

        return (
            _DefaultElement.DEFAULT_MESSAGE,
            _DefaultElement.DEFAULT_HISTORY,
            _DefaultElement.PULL_MODEL_SCUCCESS_STATUS,
            model,
        )

    def _change_model(self, model: str):
        if model not in [None, ""]:
            self.pipeline.set_model_name(model)
            self.pipeline.set_model()
            self.pipeline.set_engine()
            gr.Info(f"Change model to {model}!")
        return _DefaultElement.DEFAULT_STATUS

    def _upload_document(
        self, document: list[str], list_files: list[str] | dict
    ):
        if document in [None, []]:
            if isinstance(list_files, list):
                return (list_files, _DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return list_files.get("files")
                return document
        else:
            if isinstance(list_files, list):
                return (
                    document + list_files,
                    _DefaultElement.DEFAULT_DOCUMENT,
                )
            else:
                if list_files.get("files", None):
                    return document + list_files.get("files")
                return document

    def _reset_document(self):
        self.pipeline.reset_documents()
        gr.Info("Reset all documents!")
        return (
            _DefaultElement.DEFAULT_DOCUMENT,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _show_document_btn(self, document: list[str]):
        visible = False if document in [None, []] else True
        return (gr.update(visible=visible), gr.update(visible=visible))

    def _processing_document(
        self, document: list[str], progress=gr.Progress(track_tqdm=True)
    ):
        document = document or []
        if self.host == "host.docker.internal":
            input_files = []
            for file_path in document:
                dest = os.path.join(self._data_dir, file_path.split("/")[-1])
                shutil.move(src=file_path, dst=dest)
                input_files.append(dest)
            self.pipeline.store_nodes(input_files=input_files)
        else:
            self.pipeline.store_nodes(input_files=document)
        self.pipeline.set_chat_mode()
        gr.Info("Processing Completed!")
        return _DefaultElement.COMPLETED_STATUS

    def _change_language(self, language: str):
        self.pipeline.set_language(language)
        self.pipeline.set_chat_mode()
        gr.Info(f"Change language to {language}")

    def _undo_chat(self, history: list[list[str, str]]):
        if len(history) > 0:
            history.pop(-1)
            return history
        return _DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        self.pipeline.reset_conversation()
        gr.Info("Reset chat!")
        return (
            _DefaultElement.DEFAULT_MESSAGE,
            _DefaultElement.DEFAULT_HISTORY,
            _DefaultElement.DEFAULT_DOCUMENT,
            _DefaultElement.DEFAULT_STATUS,
        )

    def _clear_chat(self):
        self.pipeline.clear_conversation()
        gr.Info("Clear chat!")
        return (
            _DefaultElement.DEFAULT_MESSAGE,
            _DefaultElement.DEFAULT_HISTORY,
            _DefaultElement.DEFAULT_STATUS,
        )

    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        for m in self._llm_response.yield_welcome_string():
            yield m

    def build(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=_JS_LIGHT_THEME,
            css=_CSS,
            fill_height=True,
            fill_width=True,
        ) as demo:
            gr.Markdown("## Local RAG Chatbot 🤖")

            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)

                with gr.Row(variant=self._variant, equal_height=False):

                    with gr.Column(
                        variant=self._variant,
                        scale=10,
                        visible=sidebar_state.value,
                    ) as setting:
                        status = gr.Textbox(
                            label="Status",
                            value="Ready!",
                            interactive=False,
                        )
                        language = gr.Radio(
                            label="Language",
                            choices=["eng", "cs", "vi"],
                            value="eng",
                            interactive=True,
                        )
                        model = gr.Dropdown(
                            label="Choose Model:",
                            choices=[
                                "gpt-4o-mini",
                                "llama3-chatqa:8b-v1.5-q8_0",
                                "llama3-chatqa:8b-v1.5-q6_K",
                            ],
                            value=None,
                            interactive=True,
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            pull_btn = gr.Button(
                                value="Pull Model",
                                visible=False,
                                min_width=50,
                            )
                            cancel_btn = gr.Button(
                                value="Cancel",
                                visible=False,
                                min_width=50,
                            )

                        documents = gr.Files(
                            label="Add Documents",
                            value=[],
                            file_types=[".txt", ".pdf", ".csv"],
                            file_count="multiple",
                            height=150,
                            interactive=True,
                        )
                        with gr.Row():
                            upload_doc_btn = gr.UploadButton(
                                label="Upload",
                                value=[],
                                file_types=[".txt", ".pdf", ".csv"],
                                file_count="multiple",
                                min_width=20,
                                visible=False,
                            )
                            reset_doc_btn = gr.Button(
                                "Reset", min_width=20, visible=False
                            )

                    with gr.Column(scale=30, variant=self._variant):
                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant=self._variant):
                            chat_mode = gr.Dropdown(
                                choices=["chat", "QA"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=_DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row(variant=self._variant):
                            ui_btn = gr.Button(
                                value=(
                                    "Hide Setting"
                                    if sidebar_state.value
                                    else "Show Setting"
                                ),
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)

                    with gr.Column(scale=10, variant=self._variant):
                        sources_ = gr.State(self._sources)

                        @gr.render(inputs=sources_)
                        def render_sources(sources):
                            boxes = []
                            a = 1
                            for source in sources:
                                box = gr.Textbox(
                                    value=source,
                                    key=a,
                                    label=f"Box {a}",
                                    max_lines=5,
                                )
                                boxes.append(box)
                                a += 1

            with gr.Tab("Output"):
                with gr.Row(variant=self._variant):
                    log = gr.Code(
                        label="",
                        language="markdown",
                        interactive=False,
                        lines=30,
                    )
                    demo.load(
                        self.logger.read_logs,
                        outputs=[log],
                        every=1,
                        show_progress="hidden",
                        scroll_to_output=True,
                    )

            clear_btn.click(
                self._clear_chat, outputs=[message, chatbot, status]
            )
            cancel_btn.click(
                lambda: (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None,
                ),
                outputs=[pull_btn, cancel_btn, model],
            )
            undo_btn.click(
                self._undo_chat, inputs=[chatbot], outputs=[chatbot]
            )
            reset_btn.click(
                self._reset_chat,
                outputs=[message, chatbot, documents, status],
            )
            pull_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)),
                outputs=[pull_btn, cancel_btn],
            ).then(
                self._pull_model,
                inputs=[model],
                outputs=[message, chatbot, status, model],
            ).then(
                self._change_model, inputs=[model], outputs=[status]
            )
            message.submit(
                self._upload_document,
                inputs=[documents, message],
                outputs=[documents],
            ).then(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status],
            ).then(
                self._get_sources, inputs=None, outputs=[sources_]
            )
            language.change(self._change_language, inputs=[language])
            model.change(
                self._get_confirm_pull_model,
                inputs=[model],
                outputs=[pull_btn, cancel_btn, status],
            )
            documents.change(
                self._processing_document,
                inputs=[documents],
                outputs=[status],
            ).then(
                self._show_document_btn,
                inputs=[documents],
                outputs=[upload_doc_btn, reset_doc_btn],
            )
            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            upload_doc_btn.upload(
                self._upload_document,
                inputs=[documents, upload_doc_btn],
                outputs=[documents, upload_doc_btn],
            )
            reset_doc_btn.click(
                self._reset_document,
                outputs=[documents, upload_doc_btn, reset_doc_btn],
            )
            demo.load(self._welcome, outputs=[message, chatbot, status])

        return demo
