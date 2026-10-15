"""ARIFLAME: модель качества для тяжёлых творческих скиллов.

Дешёвая модель бокса (luna) хорошо болтает, но карусели, презентации и сайты
делает плохо. Наши скиллы такого рода помечены в шапке ``model: quality``
(и, по желанию, ``effort: low|medium|high``). Агент открыл такой скилл через
``skill_view`` — остаток прогона ведёт модель качества (``EA_QUALITY_MODEL``),
и следующие ходы той же сессии начинаются на ней, пока не пройдёт
``EA_QUALITY_STICKY_MIN`` минут с последнего открытия такого скилла
(«сделай третий слайд крупнее» — тоже работа для неё).

Точки входа (тул агента не видит, поэтому всё — из цикла и начала хода):
  * ``after_tools`` — agent/conversation_loop.py, между выполнением тулов и
    следующим вызовом модели: успешный ``skill_view`` скилла ``model: quality``
    → ``switch_model`` (меняется только имя модели);
  * ``on_turn_start`` — agent/turn_context.py: окно не истекло → сразу на
    модели качества (свежий агент API-сервера), истекло → назад на модель бокса
    (агент, закэшированный гейтвеем Telegram);
  * ``stamp_response_items`` + ``portable_history`` — agent/conversation_loop.py:
    зашифрованные рассуждения привязаны к модели, чужие новой не отдаём и снимаем
    id реплик, иначе API ищет к ним «родные» reasoning-элементы и отвечает 400.

``EA_QUALITY_MODEL`` пуст — ничего не меняется (метки моделей не ставятся,
окна нет, история едет как раньше).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_MODEL = "EA_QUALITY_MODEL"
ENV_EFFORT = "EA_QUALITY_EFFORT"
ENV_STICKY = "EA_QUALITY_STICKY_MIN"
DEFAULT_STICKY_MIN = 120.0
DEFAULT_EFFORT = "medium"

# Метка модели на элементах codex_reasoning_items / codex_message_items.
# Лежит ВНУТРИ элементов, потому что база хранит эти списки JSON-ом целиком —
# метка переживает перезапуск и свежего агента API-сервера.
STAMP = "_ea_model"

STORE_NAME = "ariflame_quality.json"

# Сессии чужих чатов (ea-chat: гость и группа) — модель не меняем никогда.
_GUEST_PREFIXES = ("guest_", "group_")
# Личка. Группы родного гейтвея (group/forum/channel) — как чужой чат.
_DM_CHAT_TYPES = ("", "dm", "private")
# Где подменять имя модели безопасно: тот же адрес, тот же ключ, тот же протокол.
_SWITCHABLE_API_MODES = ("codex_responses", "chat_completions")

_store_lock = threading.Lock()


# ── настройки ────────────────────────────────────────────────────────────────

def quality_model() -> str:
    return (os.getenv(ENV_MODEL) or "").strip()


def enabled() -> bool:
    return bool(quality_model())


def _sticky_seconds() -> float:
    raw = (os.getenv(ENV_STICKY) or "").strip()
    try:
        minutes = float(raw) if raw else DEFAULT_STICKY_MIN
    except ValueError:
        minutes = DEFAULT_STICKY_MIN
    return max(minutes, 0.0) * 60.0


def _effort(value: Any) -> Optional[str]:
    """Годное усилие рассуждения или None (none/false — не годится: качеству думать надо)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from hermes_constants import parse_reasoning_effort
        cfg = parse_reasoning_effort(value)
    except Exception:
        return None
    if cfg and cfg.get("enabled") and cfg.get("effort"):
        return str(cfg["effort"])
    return None


def _default_effort() -> str:
    return _effort(os.getenv(ENV_EFFORT)) or DEFAULT_EFFORT


# ── кто может переключаться ─────────────────────────────────────────────────

def session_key(agent) -> str:
    """Ключ окна: стабильный ключ чата гейтвея, иначе id сессии (API-сервер, CLI)."""
    return str(
        getattr(agent, "_gateway_session_key", None)
        or getattr(agent, "session_id", None)
        or ""
    )


def eligible(agent) -> bool:
    """Свой человек в своей сессии. Сабагенты, фоновые ревью/куратор и чужие чаты — нет."""
    if getattr(agent, "_delegate_depth", 0):
        return False
    if getattr(agent, "_memory_write_origin", None) == "background_review":
        return False
    if getattr(agent, "platform", None) == "curator":
        return False
    for value in (getattr(agent, "session_id", None), getattr(agent, "_gateway_session_key", None)):
        if str(value or "").startswith(_GUEST_PREFIXES):
            return False
    if str(getattr(agent, "_chat_type", "") or "").strip().lower() not in _DM_CHAT_TYPES:
        return False
    return bool(session_key(agent))


def _runtime_ok(agent) -> bool:
    if (getattr(agent, "provider", "") or "").strip().lower() == "moa":
        return False
    if getattr(agent, "_fallback_activated", False):
        return False  # сейчас отвечает запасной поставщик — его модель не наша
    return getattr(agent, "api_mode", None) in _SWITCHABLE_API_MODES


# ── окно: сессия → когда последний раз открывали тяжёлый скилл ───────────────

def _store_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / STORE_NAME


def _load_store() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("ARIFLAME: окно модели качества не прочиталось", exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def remember(key: str, skill: str, effort: str, *, now: Optional[float] = None) -> None:
    if not key:
        return
    now = time.time() if now is None else now
    window = _sticky_seconds()
    with _store_lock:
        data = _load_store()
        data = {
            k: v for k, v in data.items()
            if isinstance(v, dict) and now - float(v.get("t") or 0) < window
        }
        data[key] = {"t": now, "skill": skill, "effort": effort}
        try:
            from utils import atomic_json_write
            atomic_json_write(_store_path(), data, indent=1)
        except Exception:
            logger.warning("ARIFLAME: окно модели качества не записалось", exc_info=True)


def recall(key: str, *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Запись окна, если оно ещё не истекло."""
    if not key:
        return None
    now = time.time() if now is None else now
    with _store_lock:
        entry = _load_store().get(key)
    if not isinstance(entry, dict):
        return None
    try:
        age = now - float(entry.get("t") or 0)
    except (TypeError, ValueError):
        return None
    return entry if 0 <= age < _sticky_seconds() else None


# ── переключение ────────────────────────────────────────────────────────────

def _set_effort(agent, effort: str) -> None:
    agent.reasoning_config = {"enabled": True, "effort": effort}
    rt = getattr(agent, "_primary_runtime", None)
    if isinstance(rt, dict):
        rt["reasoning_config"] = dict(agent.reasoning_config)


def _swap_model(agent, model: str) -> None:
    """Меняется только имя модели: поставщик, адрес, ключ и протокол — текущие.

    Системный промпт не выбрасываем (switch_model его обнуляет): в базе остаётся
    промпт модели бокса, в памяти — он же с исправленной строкой ``Model:``,
    как при фолбэке. Вернулись назад — промпт снова байт-в-байт как в базе.

    База на время смены отключена: switch_model пишет в неё маршрут оплаты
    (поставщик и адрес у нас те же) и заодно стирает сохранённый промпт —
    следующий ход пересобрал бы его и переписал в базе на чужую модель.
    """
    prompt = getattr(agent, "_cached_system_prompt", None)
    api_key = agent.api_key if isinstance(getattr(agent, "api_key", None), str) else ""
    session_db = getattr(agent, "_session_db", None)
    agent._session_db = None
    try:
        agent.switch_model(
            model,
            agent.provider,
            api_key=api_key,
            base_url=getattr(agent, "base_url", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    finally:
        agent._session_db = session_db
    if prompt:
        agent._cached_system_prompt = prompt
        try:
            from agent.chat_completion_helpers import rewrite_prompt_model_identity
            rewrite_prompt_model_identity(agent, agent.model, agent.provider)
        except Exception:
            logger.debug("ARIFLAME: строка Model: в промпте не переписалась", exc_info=True)


def ensure_quality(agent, effort: str, reason: str) -> bool:
    """Поставить агента на модель качества. True — модель сменилась сейчас."""
    target = quality_model()
    if not target:
        return False
    if agent.model == target:
        if getattr(agent, "_ea_quality_base", None) is not None:
            _set_effort(agent, effort)  # уже на качестве нашими руками — усилие скилла
        return False
    if not _runtime_ok(agent):
        return False
    base = {
        "model": agent.model,
        "reasoning_config": copy.deepcopy(getattr(agent, "reasoning_config", None)),
        "config_context_length": getattr(agent, "_config_context_length", None),
    }
    old = agent.model
    try:
        _swap_model(agent, target)
    except Exception:
        logger.warning("ARIFLAME: модель качества %s не включилась", target, exc_info=True)
        return False
    agent._ea_quality_base = base
    _set_effort(agent, effort)
    logger.info("quality model: %s → %s (%s, effort %s)", old, target, reason, effort)
    return True


def restore(agent, reason: str) -> bool:
    """Вернуть модель бокса. True — вернули."""
    base = getattr(agent, "_ea_quality_base", None)
    if not base:
        return False
    target = quality_model()
    if target and agent.model != target:
        # Модель уже сменили руками (/model) — это решение человека, не наше.
        agent._ea_quality_base = None
        return False
    old = agent.model
    try:
        _swap_model(agent, base["model"])
    except Exception:
        logger.warning("ARIFLAME: не вернулась модель бокса %s", base.get("model"), exc_info=True)
        return False  # метку держим: попробуем на следующем ходу
    agent._ea_quality_base = None
    saved = base.get("reasoning_config")
    agent.reasoning_config = copy.deepcopy(saved)
    rt = getattr(agent, "_primary_runtime", None)
    if isinstance(rt, dict):
        rt["reasoning_config"] = dict(saved) if saved else None
    ctx_len = base.get("config_context_length")
    if ctx_len is not None:
        # switch_model сбрасывает model.context_length из конфига — возвращаем.
        agent._config_context_length = ctx_len
        cc = getattr(agent, "context_compressor", None)
        if cc is not None:
            try:
                cc.update_model(
                    model=agent.model, context_length=ctx_len, base_url=agent.base_url,
                    api_key=agent.api_key, provider=agent.provider, api_mode=agent.api_mode,
                )
            except Exception:
                logger.debug("ARIFLAME: компрессор не принял context_length", exc_info=True)
    logger.info("quality model: %s → %s (%s)", old, agent.model, reason)
    return True


# ── точки входа ─────────────────────────────────────────────────────────────

def on_turn_start(agent) -> None:
    """Начало хода: окно открыто — на модели качества, истекло — назад."""
    if not enabled() or not eligible(agent):
        if getattr(agent, "_ea_quality_base", None):
            restore(agent, "off")
        return
    entry = recall(session_key(agent))
    if entry:
        effort = _effort(entry.get("effort")) or _default_effort()
        ensure_quality(agent, effort, f"sticky, skill {entry.get('skill') or '?'}")
    elif getattr(agent, "_ea_quality_base", None):
        restore(agent, "window expired")


def _tc_field(tc: Any, *path: str) -> Any:
    cur = tc
    for name in path:
        cur = cur.get(name) if isinstance(cur, dict) else getattr(cur, name, None)
        if cur is None:
            return None
    return cur


def _tool_results(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Результаты тулов последней пачки (хвост списка до реплики ассистента).

    Между ними бывают служебные user-сообщения (картинка из результата тула) —
    их пропускаем, а не обрываем поиск.
    """
    out: Dict[str, Any] = {}
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            break
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid not in out:
            out[str(tcid)] = msg.get("content")
    return out


def _skill_frontmatter(payload: Dict[str, Any]) -> Dict[str, Any]:
    from agent.skill_utils import parse_frontmatter

    skill_dir = payload.get("skill_dir")
    if isinstance(skill_dir, str) and skill_dir:
        try:
            fm, _ = parse_frontmatter((Path(skill_dir) / "SKILL.md").read_text(encoding="utf-8"))
            return fm if isinstance(fm, dict) else {}
        except Exception:
            pass
    content = payload.get("content")
    if isinstance(content, str):
        try:
            fm, _ = parse_frontmatter(content)
            return fm if isinstance(fm, dict) else {}
        except Exception:
            return {}
    return {}


def quality_skill_from_result(content: Any) -> Optional[Tuple[str, Optional[str]]]:
    """(имя скилла, effort из шапки) — если это успешный показ SKILL.md с ``model: quality``."""
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return None
    try:
        # raw_decode: к результату мог прилипнуть хвост /steer — разбираем только JSON.
        payload, _ = json.JSONDecoder().raw_decode(content.lstrip())
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    if payload.get("file"):
        return None  # показ файла скилла, не SKILL.md
    fm = _skill_frontmatter(payload)
    if str(fm.get("model") or "").strip().lower() != "quality":
        return None
    return str(payload.get("name") or fm.get("name") or "?"), _effort(fm.get("effort"))


def after_tools(agent, assistant_message: Any, messages: List[Dict[str, Any]]) -> bool:
    """После пачки тулов: открыт SKILL.md скилла ``model: quality`` → модель качества.

    Считается только показ самого SKILL.md (без file_path): модель всегда
    начинает с него, а файлы скилла читает уже на нужной модели.
    True — модель сменилась сейчас (цикл обновит системный промпт хода).
    """
    if not enabled() or not eligible(agent):
        return False
    calls = [
        tc for tc in (getattr(assistant_message, "tool_calls", None) or [])
        if _tc_field(tc, "function", "name") == "skill_view"
    ]
    if not calls:
        return False
    results = _tool_results(messages)
    hit = None
    for tc in calls:
        raw_args = _tc_field(tc, "function", "arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except ValueError:
            args = {}
        if isinstance(args, dict) and args.get("file_path"):
            continue
        hit = quality_skill_from_result(results.get(str(_tc_field(tc, "id") or "")))
        if hit:
            break
    if not hit:
        return False
    skill, skill_effort = hit
    effort = skill_effort or _default_effort()  # скиллу виднее, потом EA_QUALITY_EFFORT
    switched = ensure_quality(agent, effort, f"skill {skill}")
    if getattr(agent, "_ea_quality_base", None) is not None:
        remember(session_key(agent), skill, effort)
    return switched


# ── история: чужие рассуждения новой модели не отдаём ────────────────────────

def stamp_response_items(agent, response: Any) -> None:
    """Метка модели, выпустившей элементы ответа (только при включённой фиче).

    Ставится сразу после нормализации ответа: реплика собирается из тех же
    словарей, поэтому метка доезжает и в историю, и в базу (реплика с вызовом
    тула пишется в базу ещё до выполнения тула).
    """
    if not enabled():
        return
    model = str(getattr(agent, "model", "") or "")
    if not model:
        return
    for key in ("codex_reasoning_items", "codex_message_items"):
        items = response.get(key) if isinstance(response, dict) else getattr(response, key, None)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item[STAMP] = model


def _stamp_of(*lists: Any) -> Optional[str]:
    for items in lists:
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get(STAMP):
                    return str(item[STAMP])
    return None


def _configured_model() -> str:
    try:
        from hermes_cli.config import load_config_readonly
        model_cfg = (load_config_readonly() or {}).get("model")
    except Exception:
        return ""
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    return ""


def _legacy_owner(agent) -> str:
    """Чьи элементы без метки (выпущены до включения фичи): модели бокса."""
    base = getattr(agent, "_ea_quality_base", None)
    if isinstance(base, dict) and base.get("model"):
        return str(base["model"])
    return _configured_model()


def _without(item: Any, keys: Iterable[str]) -> Any:
    if not isinstance(item, dict):
        return item
    drop = set(keys)
    return {k: v for k, v in item.items() if k not in drop}


def portable_history(agent, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Сообщения для запроса Responses API текущей модели.

    Реплика чужой модели: без зашифрованных рассуждений и без id у её
    message-элементов (function_call и так уходит без id). Своя — как есть,
    только без меток. Исходные сообщения не трогаем — меняем копии.
    Нет меток и фича выключена — возвращается тот же список.
    """
    on = enabled()
    current = str(getattr(agent, "model", "") or "")
    legacy_owner: Optional[str] = None
    out: Optional[List[Dict[str, Any]]] = None
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        reasoning = msg.get("codex_reasoning_items")
        items = msg.get("codex_message_items")
        if not (isinstance(reasoning, list) and reasoning) and not (isinstance(items, list) and items):
            continue
        stamp = _stamp_of(reasoning, items)
        foreign = False
        if on:
            if stamp is None:
                if legacy_owner is None:
                    legacy_owner = _legacy_owner(agent)
                foreign = bool(legacy_owner) and legacy_owner != current
            else:
                foreign = stamp != current
        if not foreign and stamp is None:
            continue
        new = dict(msg)
        if foreign:
            new.pop("codex_reasoning_items", None)
            if isinstance(items, list):
                new["codex_message_items"] = [_without(it, ("id", STAMP)) for it in items]
        else:
            if isinstance(reasoning, list):
                new["codex_reasoning_items"] = [_without(it, (STAMP,)) for it in reasoning]
            if isinstance(items, list):
                new["codex_message_items"] = [_without(it, (STAMP,)) for it in items]
        if out is None:
            out = list(messages)
        out[idx] = new
    return out if out is not None else messages
