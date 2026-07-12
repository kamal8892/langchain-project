from __future__ import annotations
import sys
from typing import Optional

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
except Exception:
    # Fallback values if colorama isn't available
    class _Fallback:
        RESET_ALL = ""
        BRIGHT = ""

    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
        MAGENTA = ""
        RESET = ""

    Style = _Fallback()


class Colors:
    DARKCYAN = Fore.CYAN
    PURPLE = Fore.MAGENTA
    YELLOW = Fore.YELLOW
    GREEN = Fore.GREEN
    RED = Fore.RED
    BOLD = Style.BRIGHT


def _print(msg: str, color: Optional[str] = None, file=sys.stdout) -> None:
    if color:
        try:
            print(f"{color}{msg}{Style.RESET_ALL}", file=file)
        except Exception:
            print(msg, file=file)
    else:
        print(msg, file=file)


def log_header(msg: str) -> None:
    _print(f"\n=== {msg} ===", Colors.GREEN)


def log_info(msg: str, color: Optional[str] = None) -> None:
    _print(f"[INFO] {msg}", color)


def log_success(msg: str, color: Optional[str] = None) -> None:
    _print(f"[SUCCESS] {msg}", color or Colors.GREEN)


def log_error(msg: str, color: Optional[str] = None) -> None:
    _print(f"[ERROR] {msg}", color or Colors.RED, file=sys.stderr)


def log_warning(msg: str, color: Optional[str] = None) -> None:
    _print(f"[WARNING] {msg}", color or Colors.YELLOW)
