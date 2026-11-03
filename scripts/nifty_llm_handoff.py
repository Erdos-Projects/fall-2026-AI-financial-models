"""Relay prompts and replies between the Nifty search runner and a chat model.

    status                                   list LLM searches and any pending prompt
    show ARM [--repeat N] [--print]          copy the pending prompt to the clipboard
    submit ARM [--repeat N] --model NAME [--file PATH] [--force]
                                             save the reply (default: clipboard) for it

Paste each prompt into a NEW temporary chat (memory and personalization off,
no browsing or code tools) and copy the model's complete reply. The runner
validates every reply and writes one repair prompt when a reply is unusable.
This tool only moves text: it never edits a prompt or replaces a saved reply.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from nifty_proposers import now
from nifty_search_space import extract_reply_json

ROOT = Path(__file__).resolve().parents[1]


def handoff_root(smoke: bool) -> Path:
    return ROOT / ("data/processed/nifty_llm_search_smoke" if smoke else "reports/nifty_llm_search") / "handoff"


def pending_number(directory: Path) -> int | None:
    numbers = sorted(int(path.name[:2]) for path in directory.glob("[0-9][0-9]_prompt.md"))
    waiting = [number for number in numbers if not (directory / f"{number:02d}_reply.txt").exists()]
    return waiting[0] if waiting else None


def status(root: Path) -> None:
    directories = sorted(path for path in root.glob("*_r*") if path.is_dir()) if root.exists() else []
    if not directories:
        print("No LLM search has written a prompt yet.")
    for directory in directories:
        number = pending_number(directory)
        messages = len(list(directory.glob("[0-9][0-9]_reply.txt")))
        state = (f"waiting for a reply to {(directory / f'{number:02d}_prompt.md').relative_to(ROOT)}"
                 if number else "no reply pending")
        print(f"{directory.name}: {messages} repl{'y' if messages == 1 else 'ies'} saved; {state}")


def show(directory: Path, print_text: bool) -> None:
    number = pending_number(directory)
    if number is None:
        raise SystemExit(f"{directory.name}: no prompt is waiting for a reply")
    prompt = directory / f"{number:02d}_prompt.md"
    text = prompt.read_text()
    if print_text or shutil.which("pbcopy") is None:
        print(text)
    else:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        print(f"Copied {prompt.relative_to(ROOT)} ({len(text):,} characters) to the clipboard.")
    print("Paste it into a NEW temporary chat, then copy the complete reply and run submit.")


def submit(directory: Path, model: str, source: Path | None, force: bool) -> None:
    number = pending_number(directory)
    if number is None:
        raise SystemExit(f"{directory.name}: no prompt is waiting for a reply")
    if source is not None:
        text = source.read_text()
    elif shutil.which("pbpaste"):
        text = subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode()
    else:
        raise SystemExit("No clipboard tool found; pass --file")
    if not text.strip():
        raise SystemExit("The reply is empty")
    if text == (directory / f"{number:02d}_prompt.md").read_text():
        raise SystemExit("The clipboard still holds the prompt; copy the model's reply instead")
    try:
        extract_reply_json(text)
    except ValueError as error:
        if not force:
            raise SystemExit(f"{error}. Check what was copied; use --force to submit an unusable reply "
                             "as the model's genuine answer (the runner then requests one repair).") from None
    meta_path = directory / f"{number:02d}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(model=model, reply_submitted_at_utc=now(), reply_chars=len(text))
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    # Write then rename, so a waiting runner never reads a partial reply.
    reply = directory / f"{number:02d}_reply.txt"
    temporary = reply.with_name(f".{reply.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(reply)
    print(f"Saved {reply.relative_to(ROOT)} ({len(text):,} characters, model: {model}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="use the smoke-test study")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    for name in ["show", "submit"]:
        command = commands.add_parser(name)
        command.add_argument("arm", choices=["llm_batch", "llm_feedback"])
        command.add_argument("--repeat", type=int, default=1)
        if name == "show":
            command.add_argument("--print", action="store_true", help="print instead of copying")
        else:
            command.add_argument("--model", required=True, help="model and interface, e.g. 'GPT-5 (ChatGPT)'")
            command.add_argument("--file", type=Path, help="read the reply from a file instead of the clipboard")
            command.add_argument("--force", action="store_true", help="submit a reply without usable JSON")
    args = parser.parse_args()
    root = handoff_root(args.smoke)
    if args.command == "status":
        status(root)
        return
    directory = root / f"{args.arm}_r{args.repeat}"
    if args.command == "show":
        show(directory, args.print)
    else:
        submit(directory, args.model, args.file, args.force)


if __name__ == "__main__":
    main()
