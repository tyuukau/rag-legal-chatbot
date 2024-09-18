import argparse
import llama_index
from dotenv import load_dotenv

from .ui import LocalChatbotApp
from .pipeline import LocalRAGPipeline
from .logger import Logger

# from .ollama import run_ollama_server, is_port_open

from .testing import mass_test


def main():

    load_dotenv()

    # CONSTANTS
    LOG_FILE = "logging.log"
    AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Set host to local or in docker container",
    )
    parser.add_argument(
        "--share", action="store_true", help="Share gradio app"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "test"],
        default="run",
        help="Specify the mode to run the script ('run' for normal execution, 'test' for testing)",
    )

    parser.add_argument(
        "--input_json",
        type=str,
        help="Path to the input JSON file for testing",
        default="data/test_questions.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to the output JSON file for testing results",
        default="data/test_results.json",
    )
    args = parser.parse_args()

    if args.mode == "test":
        if not args.input_json or not args.output_json:
            parser.error(
                "--input_json and --output_csv are required when mode is 'test'"
            )
        try:
            mass_test(args.input_json, args.output_json)
        except Exception as e:
            print(f"Error during mass test: {e}")
    else:
        # OLLAMA SERVER
        # if args.host != "host.docker.internal":
        #     port_number = 11434
        #     if not is_port_open(port_number):
        #         run_ollama_server()

        # LOGGER
        llama_index.core.set_global_handler("simple")
        logger = Logger(LOG_FILE)
        logger.reset_logs()

        # PIPELINE
        pipeline = LocalRAGPipeline(host=args.host)

        # UI
        ui = LocalChatbotApp(
            pipeline=pipeline,
            logger=logger,
            host=args.host,
            avatar_images=AVATAR_IMAGES,
        )

        if not ui.pipeline.check_store_exists():
            print("Begin ingesting data...")
            ui.ingest_data()
            print("Finished ingesting data.")

        print("Setting chat engine")
        ui.pipeline.set_chat_engine()

        print("Building UI")
        ui.build_ui().launch(
            share=args.share,
            server_name="0.0.0.0",
            debug=False,
            show_api=False,
        )


if __name__ == "__main__":
    main()
