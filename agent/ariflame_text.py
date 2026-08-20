"""ARIFLAME: user-facing service strings for our fleet.

Upstream resolves the interface language as env > ``display.language`` >
``"en"``.  Our boxes never set either, so every service string Hermes shows a
client — "Queued for the next turn", "Couldn't deliver the file attachment" —
arrived in English in a Russian conversation.  A client wrote "ничего не
понимаю".

We do not own ``config.yaml`` on the boxes, so this module keeps upstream's
resolution order and only changes the last step: when nothing is configured,
fall back to Russian instead of English.  An operator who sets
``HERMES_LANGUAGE`` or ``display.language`` still wins, so a non-Russian
client is one setting away.

Strings themselves live in ``locales/<lang>.yaml`` under the ``ariflame.``
namespace — same catalog, same loader, same fallback chain as upstream.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# What our fleet speaks when nobody said otherwise.
ARIFLAME_DEFAULT_LANGUAGE = "ru"


@lru_cache(maxsize=1)
def _explicit_config_language() -> str:
    """``display.language`` as literally written in config.yaml, or "".

    Deliberately NOT ``load_config()``: that merges the shipped defaults, and
    ``display.language: en`` is one of them — so every box looks like it was
    explicitly set to English when nobody ever chose anything.  A key present
    in the file is a decision; an absent key is not.
    """
    try:
        import yaml
        from hermes_cli.config import get_config_path

        path = get_config_path()
        if not path or not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        value = ((raw.get("display") or {}) if isinstance(raw, dict) else {}).get("language")
        return str(value).strip() if value else ""
    except Exception:
        logger.debug("ARIFLAME: could not read display.language", exc_info=True)
        return ""


def ariflame_language() -> str:
    """Resolve the language for our service strings (env > config file > ru)."""
    try:
        from agent.i18n import _normalize_lang

        env_lang = os.environ.get("HERMES_LANGUAGE")
        if env_lang:
            return _normalize_lang(env_lang)
        cfg_lang = _explicit_config_language()
        if cfg_lang:
            return _normalize_lang(cfg_lang)
    except Exception:
        logger.debug("ARIFLAME: language resolution failed", exc_info=True)
    return ARIFLAME_DEFAULT_LANGUAGE


def at(key: str, **format_kwargs: Any) -> str:
    """Translate one of our ``ariflame.*`` keys for the active language.

    Returns ``""`` when the key is not in the catalog, so callers can fall
    back to something readable.  Upstream's ``t()`` returns the KEY ITSELF on
    a miss ("ariflame.working.minutes") — that is right for a developer
    reading a log and wrong for the only audience these strings have: a
    client, in chat, who would see a dotted path where a sentence belongs.
    A miss is not a rare theoretical case either — it is exactly what a
    half-deployed catalog looks like (locales/*.yaml is copied by
    apply-fork.sh as a separate file from the code that reads it).
    """
    try:
        from agent.i18n import t

        value = t(key, lang=ariflame_language(), **format_kwargs)
    except Exception:
        logger.debug("ARIFLAME: i18n lookup failed for %s", key, exc_info=True)
        return ""
    if not value or value == key:
        logger.warning("ARIFLAME: ключа %s нет в каталоге", key)
        return ""
    return value


__all__ = ["at", "ariflame_language", "ARIFLAME_DEFAULT_LANGUAGE"]
