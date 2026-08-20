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
from typing import Any

logger = logging.getLogger(__name__)

# What our fleet speaks when nobody said otherwise.
ARIFLAME_DEFAULT_LANGUAGE = "ru"


def ariflame_language() -> str:
    """Resolve the language for our service strings (env > config > ru)."""
    try:
        from agent.i18n import _config_language_cached, _normalize_lang

        env_lang = os.environ.get("HERMES_LANGUAGE")
        if env_lang:
            return _normalize_lang(env_lang)
        cfg_lang = _config_language_cached()
        if cfg_lang:
            return cfg_lang
    except Exception:
        logger.debug("ARIFLAME: language resolution failed", exc_info=True)
    return ARIFLAME_DEFAULT_LANGUAGE


def at(key: str, **format_kwargs: Any) -> str:
    """Translate one of our ``ariflame.*`` keys for the active language."""
    try:
        from agent.i18n import t

        return t(key, lang=ariflame_language(), **format_kwargs)
    except Exception:
        logger.debug("ARIFLAME: i18n lookup failed for %s", key, exc_info=True)
        return ""


__all__ = ["at", "ariflame_language", "ARIFLAME_DEFAULT_LANGUAGE"]
