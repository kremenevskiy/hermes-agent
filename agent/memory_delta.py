"""ARIFLAME: доставка свежей памяти модели в том же ходу.

**Что было.** ``MemoryStore.format_for_system_prompt()`` отдаёт снимок,
снятый один раз в ``load_from_disk()``, а системный промпт хранится в
``sessions.system_prompt`` и переписывается ровно в трёх случаях: первый ход
сессии, компакция, смена модели или провайдера. Значит запись, сделанная в
середине темы, до модели не доезжает — иногда сутками. Замер по живому
проду: 252 из 299 успешных записей отсутствуют в системном промпте той
сессии, где они сделаны. Человек говорит правило, агент отвечает
«запомнила», кладёт факт на диск — и через час его не знает. Это худший из
возможных отказов, потому что он выглядит как успех.

**Почему нельзя чинить в лоб.** Пересобирать системный промпт каждый ход —
значит рвать префикс-кэш каждый ход. На тяжёлой ветке это сотни тысяч
токенов по полной ставке вместо кэша, то есть реальные деньги на каждом
сообщении. Замороженный снимок придуман правильно; сломан не он.

**Что сделано.** Дельта — разница между тем, что на диске, и тем, что в
промпте, — едет по тому же эфемерному каналу, что и живая дата и заметки
гейтвея: приклеивается к сообщению пользователя, попадает в САМЫЙ ХВОСТ
запроса, за кэшируемым префиксом. Префикс не двигается, платим только за
несколько десятков токенов дельты, и то один раз на запись.

**Один раз — как это устроено.** У каждой записи есть короткий
идентификатор ``m:xxxxxx`` (см. ``tools/memory_tool.entry_marker``). Прежде
чем довозить запись, смотрим, нет ли её идентификатора в уже отправленных
сообщениях пользователя: канал ``api_content`` переигрывается байт-в-байт,
поэтому доставленное однажды остаётся в контексте и повторять его незачем.
После компакции история короче — маркер пропадает, запись доезжает заново,
и специально ловить компакцию не нужно. А как только промпт пересобран со
свежей памятью, дельта становится пустой сама: блок исчезает, цена
возвращается к нулю.

**Что ещё в блоке.** Записи, которые из памяти УДАЛИЛИ или переписали, в
замороженном промпте всё ещё стоят как есть. Модель их видит и считает
действующими — правка «теперь вертикальные, а не квадратные» до неё не
доходит, зато старое правило доходит прекрасно. Поэтому вторая половина
блока называет такие строки недействительными.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Потолки. Дельта — это хвост запроса, который платится по полной ставке,
# поэтому она обязана быть маленькой. Шесть записей это перекрывает любой
# нормальный ход (в живых данных за ход пишется 1-2 записи); всё, что не
# влезло, честно считается и доедет после компакции вместе с промптом.
_MAX_FRESH = 6
_MAX_STALE = 3
_MAX_ENTRY_CHARS = 400
_MAX_BLOCK_CHARS = 2000
# Сколько последних сообщений просматриваем в поисках уже доставленных
# маркеров. Больше не нужно: маркер, ушедший 200 сообщений назад, всё равно
# не переживёт ближайшую компакцию.
_SCAN_LAST_MESSAGES = 200

_SEPARATOR = "═" * 46


def _block_bounds(prompt: str, header: str) -> Optional[Tuple[int, int]]:
    """Границы содержимого одного блока памяти внутри системного промпта.

    Блок собирается в ``MemoryStore._render_block`` как «разделитель,
    заголовок, разделитель, содержимое», а блоки склеиваются через пустую
    строку. Значит содержимое кончается там, где начинается следующий
    разделитель, либо в конце строки.
    """
    at = prompt.find(header)
    if at < 0:
        return None
    body_start = prompt.find("\n", at)
    if body_start < 0:
        return None
    # После заголовка идёт закрывающий разделитель — перешагнуть его.
    if prompt[body_start + 1:].startswith(_SEPARATOR):
        body_start = prompt.find("\n", body_start + 1)
        if body_start < 0:
            return None
    body_start += 1
    nxt = prompt.find(_SEPARATOR, body_start)
    end = nxt if nxt >= 0 else len(prompt)
    # Промпты, собранные ДО появления закрывающего разделителя, лежат в
    # ``sessions.system_prompt`` живых веток и будут лежать там до ближайшей
    # компакции. У них блок кончается ничем, и без этой отсечки в «записи»
    # попал бы хвост промпта — а он выглядел бы как снятое правило.
    legacy = prompt.find("\nConversation started:", body_start)
    if 0 <= legacy < end:
        end = legacy
    return (body_start, end)


def parse_prompt_entries(prompt: str) -> Dict[str, List[str]]:
    """Достать из системного промпта записи памяти, которые модель ВИДИТ.

    Сравнивать надо именно с промптом, а не со снимком в объекте стора:
    гейтвей строит агента заново на каждый ход, и снимок в новом объекте
    свежий, тогда как промпт достаётся из базы дословно и может быть
    недельной давности. Разошлись именно эти двое.
    """
    from tools.memory_tool import ENTRY_DELIMITER, MEMORY_BLOCK_HEADERS

    out: Dict[str, List[str]] = {"memory": [], "user": []}
    if not prompt:
        return out
    for target, header in MEMORY_BLOCK_HEADERS.items():
        bounds = _block_bounds(prompt, header)
        if not bounds:
            continue
        body = prompt[bounds[0]:bounds[1]].strip()
        if not body:
            continue
        out[target] = [e.strip() for e in body.split(ENTRY_DELIMITER) if e.strip()]
    return out


def _delivered_markers(messages: Sequence[Any]) -> str:
    """Склейка недавних ПОЛЬЗОВАТЕЛЬСКИХ сообщений — там живут наши блоки.

    Только роль ``user``: ответ тула тоже содержит идентификатор записи
    (мы его туда кладём осознанно), и если считать его доставкой, то запись
    никогда не доедет — тот самый случай, когда защита от повтора отменяет
    саму работу.
    """
    chunks: List[str] = []
    for msg in list(messages or [])[-_SCAN_LAST_MESSAGES:]:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for key in ("api_content", "content"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                chunks.append(value)
    return "\n".join(chunks)


def _clip(entry: str) -> str:
    flat = entry.strip()
    if len(flat) > _MAX_ENTRY_CHARS:
        flat = flat[:_MAX_ENTRY_CHARS].rstrip() + "…"
    return flat


def compute_delta(
    live: Dict[str, List[str]],
    prompt_entries: Dict[str, List[str]],
    delivered: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], int]:
    """Что модель ещё не знает и что знает неверно.

    Возвращает (свежие, недействительные, сколько свежих не поместилось).
    Свежая запись — та, которой нет ни в промпте, ни среди уже доставленных
    маркеров. Недействительная — та, которая в промпте есть, а в памяти её
    уже нет.
    """
    from tools.memory_tool import entry_marker

    fresh: List[Tuple[str, str]] = []
    stale: List[Tuple[str, str]] = []

    for target in ("memory", "user"):
        in_prompt = set(prompt_entries.get(target) or ())
        live_entries = list(live.get(target) or ())
        for entry in live_entries:
            if not entry or entry in in_prompt:
                continue
            marker = entry_marker(entry)
            if marker in delivered:
                continue
            fresh.append((target, entry))

        live_set = set(live_entries)
        for entry in prompt_entries.get(target) or ():
            # Заблокированные сканером записи в промпте заменены заглушкой —
            # они и не должны совпадать с живыми, это не «удалили».
            if entry.startswith("[BLOCKED:") or entry in live_set:
                continue
            # Снятое правило тоже объявляется ОДИН раз: без этого «это больше
            # не так» ехало бы в каждом ходу до самой компакции — та же
            # утечка, от которой мы избавляемся у свежих записей.
            if entry_marker(entry) in delivered:
                continue
            stale.append((target, entry))

    # Свежими считаем последние: файл памяти всегда дописывается в конец,
    # значит хвост — это то, что сказали только что, а голова — то, что
    # доедет с ближайшей компакцией.
    overflow = max(0, len(fresh) - _MAX_FRESH)
    fresh = fresh[-_MAX_FRESH:]

    # Недействительных бывает много ровно в одном случае: промпт вообще от
    # другой эпохи (сессия старая, память с тех пор переписана целиком).
    # Перечислять сотню строк «это больше не так» дороже, чем промолчать и
    # дождаться компакции, которая перепишет промпт целиком.
    if len(stale) > _MAX_STALE:
        stale = []

    return fresh, stale, overflow


def render_block(
    fresh: List[Tuple[str, str]], stale: List[Tuple[str, str]], overflow: int
) -> str:
    """Собрать текст блока. Пустая дельта — пустая строка и ноль токенов."""
    from tools.memory_tool import entry_marker

    if not fresh and not stale:
        return ""

    lines = [
        "<memory-updates>",
        "[System note: these lines were saved to your memory AFTER the system "
        "prompt above was built, so the MEMORY / USER PROFILE blocks up there do "
        "not have them yet. They are current and they win. Do not save them "
        "again. To edit one, call memory with old_text set to its id.]",
    ]
    for target, entry in fresh:
        label = "USER" if target == "user" else "MEMORY"
        lines.append(f"{label} {entry_marker(entry)} | {_clip(entry)}")
    if overflow:
        lines.append(
            f"[+{overflow} more entries were saved and are on disk; they will "
            f"appear in the prompt itself after the next context compaction.]"
        )
    if stale:
        lines.append(
            "[These lines are still printed in the memory blocks above but were "
            "removed or rewritten since — treat them as no longer true:]"
        )
        for target, entry in stale:
            label = "USER" if target == "user" else "MEMORY"
            lines.append(f"{label} {entry_marker(entry)} (dropped) | {_clip(entry)}")
    lines.append("</memory-updates>")

    block = "\n".join(lines)
    if len(block) > _MAX_BLOCK_CHARS:
        block = block[:_MAX_BLOCK_CHARS].rstrip() + "\n…\n</memory-updates>"
    return block


def build_memory_delta_block(
    agent: Any, active_system_prompt: Optional[str], messages: Sequence[Any]
) -> str:
    """Точка входа для ``build_turn_context``. Никогда не бросает.

    Память — это довесок к ходу, а не ход. Любая ошибка здесь должна стоить
    ровно одного WARNING в логе и ни одного сорванного ответа человеку.
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return ""
    try:
        live: Dict[str, List[str]] = {}
        # Уважаем те же выключатели, что и сборка системного промпта: если
        # хранилище выключено, довозить из него нечего.
        if getattr(agent, "_memory_enabled", True):
            live["memory"] = store.live_entries("memory")
        if getattr(agent, "_user_profile_enabled", True):
            live["user"] = store.live_entries("user")
        if not any(live.values()):
            return ""

        # Тот же санитайзер, что и у снимка промпта: запись с промптовой
        # инъекцией не имеет права въехать в запрос через боковую дверь
        # только потому, что она свежая.
        for target, entries in list(live.items()):
            filename = "USER.md" if target == "user" else "MEMORY.md"
            cleaned = store._sanitize_entries_for_snapshot(entries, filename)
            live[target] = [e for e in cleaned if e and not e.startswith("[BLOCKED:")]

        prompt_entries = parse_prompt_entries(active_system_prompt or "")
        delivered = _delivered_markers(messages)
        fresh, stale, overflow = compute_delta(live, prompt_entries, delivered)
        block = render_block(fresh, stale, overflow)
        if block:
            logger.info(
                "ARIFLAME: доставляю в ход %s новых записей памяти и %s снятых "
                "(ещё %s ждут компакции), %s символов",
                len(fresh), len(stale), overflow, len(block),
            )
        return block
    except Exception:
        logger.warning("ARIFLAME: блок свежей памяти не собран", exc_info=True)
        return ""
