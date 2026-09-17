import argparse
import logging
from pathlib import Path

from brain_harmonization.config import load_config
from brain_harmonization.data.synthetic import create_demo
from brain_harmonization.pipeline import analyze, prepare, process, qc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="IXI T1 research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "preprocess", "qc", "analyze", "run"]:
        child = sub.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
    demo = sub.add_parser("demo", help="Generate artificial images and run a software smoke test")
    demo.add_argument("--data", type=Path, default=Path("demo-data"))
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--subjects", type=int, default=60)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "demo":
        args.config = create_demo(args.data, args.output, args.subjects)
        args.command = "run"
    config = load_config(args.config)
    stages = (
        [prepare, process, qc, analyze]
        if args.command == "run"
        else [
            {"prepare": prepare, "preprocess": process, "qc": qc, "analyze": analyze}[args.command]
        ]
    )
    handler = None
    try:
        for stage in stages:
            if stage is not prepare and handler is None:
                log_path = Path(config["output"]) / "run.log"
                if not log_path.parent.is_dir():
                    raise FileNotFoundError("Run prepare first")
                handler = logging.FileHandler(log_path, encoding="utf-8")
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
                )
                logging.getLogger().addHandler(handler)
            stage(config)
    finally:
        if handler:
            logging.getLogger().removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
