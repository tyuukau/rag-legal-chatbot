import os
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
    DEFAULT_MESSAGE: str = ""
    DEFAULT_MODEL: str = "gpt-4o-mini"
    DEFAULT_HISTORY: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"


class LLMResponse:
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


class LocalChatbotApp:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        host: str = "host.docker.internal",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
    ):
        self.pipeline = pipeline
        self.logger = logger
        self.host = host
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._llm_response = LLMResponse()
        self._sources: list[str] = []

    def _change_language(self, language: str):
        self.pipeline.set_chat_engine(language=language)
        gr.Info(f"Change language to {language}")

    def _get_sources(self):
        # print("Getting sources:", self._sources)
        return self._sources

    def _get_respone(
        self,
        message: str,
        chatbot: list[list[str, str]],
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        if message in [None, ""]:
            for m in self._llm_response.yield_empty_message_string():
                yield m
            self._sources = []
        else:
            console = sys.stdout
            sys.stdout = self.logger
            response = self.pipeline.query(message, chatbot)
            for m in self._llm_response.yield_stream_response(
                message, chatbot, response
            ):
                yield m
            sys.stdout = console
            self._sources = [
                n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                for n in response.source_nodes
            ]

    async def _aget_respone(
        self,
        message: str,
        chatbot: list[list[str, str]],
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        if message in [None, ""]:
            for m in self._llm_response.yield_empty_message_string():
                yield m
            self._sources = []
        else:
            console = sys.stdout
            sys.stdout = self.logger
            response = await self.pipeline.aquery(message, chatbot)
            for m in self._llm_response.yield_stream_response(
                message, chatbot, response
            ):
                yield m
            sys.stdout = console
            self._sources = [
                n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                for n in response.source_nodes
            ]

    def _undo_chat(self, history: list[list[str, str]]):
        if len(history) > 0:
            history.pop(-1)
            return history
        return _DefaultElement.DEFAULT_HISTORY

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

    ##################
    # PUBLIC METHODS #
    ##################

    def ingest_data(self):
        print("Starting Processing...")
        self.pipeline.process_document_dir()
        self.pipeline.store_nodes()
        print("Processing Completed!")

    ######################
    # The User Interface #
    ######################

    def build_ui(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=_JS_LIGHT_THEME,
            css=_CSS,
            fill_height=True,
            fill_width=True,
        ) as demo:

            ######################
            # The UI Declaration #
            ######################

            gr.Markdown("## Local RAG Chatbot 🤖")

            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)

                with gr.Row(variant="panel", equal_height=False):

                    with gr.Column(
                        variant="panel",
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
                            value=self.pipeline._language,
                            interactive=True,
                        )

                    with gr.Column(scale=30, variant="panel"):
                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant="panel"):
                            message = gr.Textbox(
                                value=_DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row(variant="panel"):
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

                    with gr.Column(scale=10, variant="panel"):
                        sources_ = gr.State(self._sources)

                        @gr.render(inputs=sources_)
                        def render_sources(sources):
                            # print("Rendering sources:", sources)
                            a = 1
                            for source in sources:
                                gr.Textbox(
                                    value=source,
                                    label=f"Source {a}",
                                    max_lines=5,
                                )
                                a += 1

            with gr.Tab("Output"):
                with gr.Row(variant="panel"):
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

            ##################
            # The Behaviours #
            ##################

            message.submit(
                self._get_respone,
                inputs=[message, chatbot],
                outputs=[message, chatbot, status],
            ).then(self._get_sources, inputs=None, outputs=[sources_])

            language.change(self._change_language, inputs=[language])

            clear_btn.click(
                self._clear_chat, outputs=[message, chatbot, status]
            )
            undo_btn.click(
                self._undo_chat, inputs=[chatbot], outputs=[chatbot]
            )

            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )

            demo.load(self._welcome, outputs=[message, chatbot, status])

        return demo
