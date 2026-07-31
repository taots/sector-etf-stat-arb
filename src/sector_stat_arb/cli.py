"""CLI entry points."""

import argparse
from .demo import run_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download","backtest","robustness","report","demo"])
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.output)
    else:
        print("Command:", args.command)


if __name__ == "__main__":
    main()
