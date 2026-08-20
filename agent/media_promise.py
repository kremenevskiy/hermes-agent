"""ARIFLAME: catch ``MEDIA:`` promises pointing at files that do not exist.

A model can invent a filename in the exact shape of a real one.  The gateway
resolves ``MEDIA:`` paths strictly, a missing file fails to resolve, the
directive is stripped out of the text, and the drop is written to a log line
nobody reads.  Live case: the agent announced three finished pictures that had
never existed, the person got the text and zero attachments, and nothing told
them anything was wrong.

Two defences are built on this module:

* ``missing_media_paths`` runs inside the conversation loop, BEFORE the reply
  leaves the agent, so the model gets the failure back as a turn error and
  fixes it in the same message (``agent/conversation_loop.py``);
* the delivery path says so out loud when a file still could not be attached
  (``gateway/platforms/base.py``, ``gateway/run.py``).

The matcher here is deliberately NARROWER than the gateway's
``MEDIA_TAG_CLEANUP_RE``: its only job is to answer "did the model name a
local file that is not on disk".  A miss costs nothing (the old behaviour), a
false positive would burn a model turn — so it only looks at unambiguous
local paths and never at ``MEDIA:https://…`` URLs, which are delivered by URL
and never touch the filesystem.
"""

from __future__ import annotations

import os
import re
from typing import List

# Quoted path, or a path anchored at ``~/`` or ``/``.  No extension filter:
# a promise is a promise whatever the file type.
_MEDIA_LOCAL_RE = re.compile(
    r"""MEDIA:\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|(?:~/|/)[^\s`"'<>|]+)""",
    re.IGNORECASE,
)

# Fenced code blocks hold documentation and examples ("include
# MEDIA:/absolute/path/to/file"), never a real delivery.  Masked
# length-preserving so nothing else shifts.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _mask_fences(text: str) -> str:
    return _FENCE_RE.sub(lambda m: " " * (m.end() - m.start()), text)


def missing_media_paths(response: str, limit: int = 8) -> List[str]:
    """Return the local ``MEDIA:`` paths in ``response`` that are not on disk."""
    if not response or "MEDIA:" not in response:
        return []

    missing: List[str] = []
    seen = set()
    for match in _MEDIA_LOCAL_RE.finditer(_mask_fences(response)):
        raw = match.group("path").strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "`\"'":
            raw = raw[1:-1].strip()
        raw = raw.rstrip("`\"',.;:)]}")
        if not raw or raw in seen:
            continue
        seen.add(raw)
        try:
            path = os.path.expanduser(raw)
        except (OSError, RuntimeError, ValueError):
            continue
        if not os.path.isabs(path):
            continue
        try:
            exists = os.path.isfile(path)
        except OSError:
            exists = False
        if not exists:
            missing.append(raw)
            if len(missing) >= limit:
                break
    return missing


def build_media_nudge(missing: List[str]) -> str:
    """The turn error handed back to the model for the missing files."""
    listed = "\n".join("  - " + p for p in missing)
    return (
        "[System note — this is not the user speaking.]\n"
        "Your reply announces attachments that do not exist. These MEDIA: "
        "paths are not on disk:\n" + listed + "\n\n"
        "Nothing was attached. If this reply were delivered as written, the "
        "person would read that the files are ready and receive nothing, with "
        "no warning — that is the one outcome that must never happen.\n\n"
        "Fix it now, in this same reply:\n"
        "1. If you meant to produce these files, produce them — call the tool "
        "that creates them, verify the file exists, then repeat your reply "
        "with the real paths.\n"
        "2. If you cannot produce them, drop those MEDIA: lines and say "
        "plainly, in the language the user is writing in, that the file was "
        "not created and what you are doing about it.\n"
        "Never state that a file has been sent when it has not."
    )


__all__ = ["missing_media_paths", "build_media_nudge"]
