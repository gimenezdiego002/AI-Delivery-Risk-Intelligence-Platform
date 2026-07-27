"""Basic tracked-source credential check for local use and CI.

This is a narrow deployment-readiness guard, not a replacement for a dedicated
secret-scanning product or a penetration test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Google-style key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".example",
    ".dockerignore",
    ".gitignore",
}


def candidate_files() -> list[Path]:
    """Return tracked and untracked non-ignored files, excluding local secrets."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PROJECT_ROOT / item.decode()
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    """Fail when a high-confidence credential pattern appears in tracked text."""
    findings: list[str] = []
    candidates = candidate_files()
    if PROJECT_ROOT / ".env" in candidates:
        findings.append(".env is tracked by Git")

    for path in candidates:
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES
            and path.name not in {"Dockerfile", ".dockerignore", ".gitignore"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    f"{path.relative_to(PROJECT_ROOT)}: possible {label}"
                )

    if findings:
        raise SystemExit(
            "Credential check failed:\n- " + "\n- ".join(findings)
        )
    print(
        "Credential check passed across "
        f"{len(candidates)} tracked/unignored files."
    )


if __name__ == "__main__":
    main()
