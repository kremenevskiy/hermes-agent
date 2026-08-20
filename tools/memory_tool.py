#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

ARIFLAME: снимок остаётся замороженным — это правильно и это экономит деньги.
Неправильным было другое: до модели запись не доезжала ВООБЩЕ, пока не
случится компакция (замер: 252 из 299 успешных записей отсутствуют в
системном промпте той сессии, где сделаны). Разницу между тем, что на диске,
и тем, что в промпте, теперь довозит ``agent/memory_delta.py`` — по
эфемерному каналу пользовательского сообщения, за кэшируемым префиксом.
Здесь для этого есть два кирпича: ``live_entries()`` (живое состояние) и
``entry_marker()`` (устойчивый идентификатор записи).

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import difflib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional, Tuple

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

# Stable header prefixes for the system-prompt memory blocks rendered by
# MemoryStore._render_block. Exported so compression's prompt-retention check
# (agent/conversation_compression.py) can detect a leftover block for a
# target whose entries have since been emptied — keep in lockstep with
# _render_block below.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

ENTRY_DELIMITER = "\n§\n"

# ARIFLAME: адрес записи.
#
# Апстрим адресует запись подстрокой (``old_text in entry``) — точным
# посимвольным вхождением. На живых людях это давало ровно три класса
# промахов: модель начинала цитату с середины фразы и не угадывала регистр
# первой буквы («П» против «п» — два разошедшихся сообщения у одной
# клиентки), модель искала запись не в том хранилище (6 промахов из 14),
# модель повторяла тот же вызов слово в слово, потому что ошибка не говорила
# ни где искали, ни что рядом. Подстроку мы не выбрасываем — она работает и
# на неё завязан контракт тула, — а достраиваем каскадом и добавляем
# короткий устойчивый идентификатор, которым можно попасть в запись
# ОДНОЗНАЧНО. Идентификатор считается от текста, а не хранится: файл памяти
# остаётся простым списком строк через §, и никакой миграции не нужно.
_MARKER_RE = re.compile(r"^\[?\s*m:([0-9a-f]{6})\s*\]?$", re.I)

# Что считаем одним и тем же символом при нестрогом сравнении: разные тире,
# разные кавычки и разные пробелы приходят от разных клавиатур и разных
# клиентов, а человек имеет в виду один и тот же текст.
_MATCH_TRANSLATION = {
    ord("«"): '"', ord("»"): '"',      # « »
    ord("“"): '"', ord("”"): '"',      # “ ”
    ord("„"): '"', ord("‘"): "'",      # „ ‘
    ord("’"): "'", ord("–"): "-",      # ’ –
    ord("—"): "-", ord("−"): "-",      # — −
    ord(" "): " ", ord("ё"): "е",  # nbsp, ё→е
    ord("Ё"): "е",                      # Ё→е
}


def normalize_for_match(text: str) -> str:
    """Свести текст к форме, в которой сравнение не спорит о мелочах.

    Регистр, вид кавычек и тире, ё/е, кратность пробелов — всё это модель
    воспроизводит неточно, а человек различий не делает. Нормализация
    используется и для нестрогого поиска, и для идентификатора: две записи,
    отличающиеся только этим, получают один и тот же ``m:``.
    """
    if not text:
        return ""
    return " ".join(text.translate(_MATCH_TRANSLATION).casefold().split())


def entry_marker(text: str) -> str:
    """Короткий устойчивый идентификатор записи: ``m:ab12cd``.

    Считается от нормализованного текста, поэтому переживает переписывание
    регистра и пробелов, но меняется при изменении смысла — что и нужно:
    правка записи это новая запись, и её идентификатор обязан быть другим.
    Шесть шестнадцатеричных знаков — это 16 млн значений на файл в сотню
    записей; коллизия невероятна, а если случится, каскад поиска отдаст
    «несколько совпадений», а не молча испортит чужую запись.
    """
    digest = hashlib.sha1(normalize_for_match(text).encode("utf-8")).hexdigest()
    return "m:" + digest[:6]


def parse_marker(needle: str) -> Optional[str]:
    """Вернуть ``m:xxxxxx``, если строка — идентификатор, иначе None."""
    if not needle:
        return None
    m = _MARKER_RE.match(needle.strip())
    return ("m:" + m.group(1).lower()) if m else None


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


# ---------------------------------------------------------------------------
# ARIFLAME: цена неудачи.
#
# Апстрим на КАЖДУЮ неудачу возвращал модели всю память целиком в поле
# ``current_entries``. Замер по живым боксам: медиана такого ответа 20 380
# символов против 203 у успешного — провал в сто раз дороже удачи, и он
# висит в контексте до самой компакции. У одной клиентки так сожжено
# 389 542 символа. Дальше — хуже: увидев весь список, модель находит в нём
# «что бы ещё поправить» и делает лишние вызовы.
#
# Полный список нужен ровно в одном случае: записей мало и они короткие,
# тогда он и стоит копейки. Во всех остальных отдаём выжимку: сколько
# записей, сколько символов и несколько превью С ИДЕНТИФИКАТОРАМИ — по ним
# можно попасть в запись точно, чего по обрезанному превью было нельзя.
# ---------------------------------------------------------------------------

# Полный список отдаём, только если он заведомо дешевле выжимки.
_INLINE_ENTRIES_MAX = 8
_INLINE_ENTRIES_CHARS = 900
# Сколько превью показываем, когда список большой, и какой ширины.
_PREVIEW_ENTRIES = 6
_PREVIEW_WIDTH = 110


def _preview(entry: str, width: int = _PREVIEW_WIDTH) -> str:
    """Однострочное превью записи с её идентификатором."""
    flat = " ".join((entry or "").split())
    if len(flat) > width:
        flat = flat[:width].rstrip() + "…"
    return f"[{entry_marker(entry)}] {flat}"


def entries_digest(entries: List[str]) -> List[str]:
    """Что показать модели вместо всего списка записей.

    Короткий список отдаётся как есть (так дешевле и точнее), длинный —
    превью последних записей: свежие интереснее древних, а весь список
    модели для исправления одной ошибки не нужен.
    """
    entries = entries or []
    total = len(ENTRY_DELIMITER.join(entries)) if entries else 0
    if len(entries) <= _INLINE_ENTRIES_MAX and total <= _INLINE_ENTRIES_CHARS:
        return list(entries)
    shown = [_preview(e) for e in entries[-_PREVIEW_ENTRIES:]]
    hidden = len(entries) - len(shown)
    if hidden > 0:
        shown.insert(0, f"…ещё {hidden} записей не показаны (всего {len(entries)})")
    return shown


def find_entry_indices(entries: List[str], needle: str) -> Tuple[List[int], str]:
    """Найти запись по ``old_text`` каскадом от точного к нестрогому.

    Возвращает (индексы, чем совпало). Порядок ступеней — от самого
    надёжного к самому снисходительному, и первая же сработавшая ступень
    останавливает поиск: нестрогое сравнение не должно перебивать точное.

    Ступени:
      ``id``         — ``old_text`` это ``m:xxxxxx`` (однозначно);
      ``exact``      — подстрока как в апстриме;
      ``normalized`` — подстрока без учёта регистра, кавычек, тире, ё/е и
                       кратности пробелов (это тот самый промах «П» против
                       «п», который стоил клиентке двух потерянных правок);
      ``fuzzy``      — ближайшая запись по SequenceMatcher, и только если
                       она отрывается от второй заметным зазором. Без
                       зазора нестрогое совпадение опаснее отказа: молча
                       переписать не ту запись хуже, чем сказать «не нашёл».
    """
    entries = entries or []
    needle = (needle or "").strip()
    if not needle or not entries:
        return [], ""

    marker = parse_marker(needle)
    if marker:
        hits = [i for i, e in enumerate(entries) if entry_marker(e) == marker]
        return hits, "id" if hits else ""

    hits = [i for i, e in enumerate(entries) if needle in e]
    if hits:
        return hits, "exact"

    n_needle = normalize_for_match(needle)
    if not n_needle:
        return [], ""
    normalized = [normalize_for_match(e) for e in entries]
    hits = [i for i, e in enumerate(normalized) if n_needle in e]
    if hits:
        return hits, "normalized"

    # Совсем короткую цитату нечётко не ищем: на трёх символах «похоже»
    # означает «случайно».
    if len(n_needle) < 12:
        return [], ""
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, n_needle, e).ratio(), i)
            for i, e in enumerate(normalized)
        ),
        reverse=True,
    )
    best_ratio, best_idx = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_ratio >= 0.72 and (best_ratio - runner_up) >= 0.08:
        return [best_idx], "fuzzy"
    return [], ""


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # After this many failed consolidation attempts (overflow / zero-match) in
    # ONE turn, stop instructing the model to "retry in this turn" and return a
    # terminal "save skipped" result so a fragile replace/add can't loop the
    # turn to budget exhaustion and suppress the user's reply (issue #42405).
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375,
                 auto_reclaim: Optional[bool] = None,
                 auto_consolidate: Optional[bool] = None):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed at-capacity consolidation attempts; reset
        # at each turn boundary by reset_consolidation_failures() (#42405).
        self._consolidation_failures = 0
        # ARIFLAME: вытеснение (память освобождает место сама) и фоновая
        # консолидация. Ключи ``memory.auto_reclaim`` / ``memory.auto_consolidate``,
        # по умолчанию включены: выключенными они означают ровно то поведение,
        # из-за которого три бокса неделю ничего не запоминали.
        self._auto_reclaim = auto_reclaim
        self._auto_consolidate = auto_consolidate

    # ── ARIFLAME: публичный доступ к живому состоянию ──
    # Фоновой консолидации и сборщику дельты для промпта нужны записи и
    # лимиты. Лезть в ``_entries_for`` снаружи — значит завязаться на
    # приватное имя из четырёх мест; лучше назвать это интерфейсом один раз.

    def live_entries(self, target: str) -> List[str]:
        """Живые записи хранилища (не снимок промпта)."""
        return list(self._entries_for(target))

    def char_limit(self, target: str) -> int:
        return self._char_limit(target)

    def _flag(self, name: str) -> bool:
        """Значение выключателя, прочитанное из конфига ОДИН раз.

        Читается лениво и запоминается на экземпляре: стор живёт столько же,
        сколько агент, а перечитывать YAML на каждую запись — это дисковый
        ввод-вывод под файловым локом ради значения, которое не меняется.
        Конфиг доедет со следующим экземпляром агента, как и все остальные
        ключи ``memory.*``.
        """
        attr = f"_{name}"
        cached = getattr(self, attr, None)
        if cached is not None:
            return bool(cached)
        try:
            from hermes_cli.config import load_config

            value = ((load_config() or {}).get("memory") or {}).get(name, True)
        except Exception:
            value = True  # конфига нет — работаем, а не молчим
        value = bool(value)
        setattr(self, attr, value)
        return value

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count an at-capacity consolidation failure and degrade gracefully.

        Under the per-turn cap, return ``response`` unchanged (it already tells
        the model how to self-correct + retry in this turn). Once the cap is
        exceeded, drop the retry instruction and return a TERMINAL result so the
        model stops looping memory calls and proceeds to answer the user — a
        failed memory side effect must never block the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        see poisoned entries by inspecting the source files directly, and remove them — silently dropping them would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text so the user
        # can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str, *, skip_drift: bool = False) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.

        When *skip_drift* is True the round-trip / entry-size check is
        bypassed.  Used by the ``add`` action which appends without
        rewriting, so existing content is never clobbered.
        """
        path = self._path_for(target)
        bak = None if skip_drift else self._detect_external_drift(target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    # ── ARIFLAME: место под новую запись ──────────────────────────────

    def _archive_path(self, target: str) -> Path:
        """Файл, куда уезжают вытесненные записи.

        Рядом с хранилищем и с тем же расширением — чтобы человек нашёл его
        глазами, а не по документации. Рантайм его не читает: это не третье
        хранилище, а корзина, из которой можно достать руками.
        """
        path = self._path_for(target)
        return path.with_name(path.stem + ".archive.md")

    def _archive(self, target: str, entries: List[str]) -> None:
        """Дописать вытесненные записи в архив. Никогда не бросает.

        Вытеснение обязано быть обратимым: запись уходит из промпта, но не с
        диска. Если архив не записался — вытеснение всё равно состоялось, и
        честнее громко сказать об этом в лог, чем отменить сохранение нового
        факта (ради которого всё и затевалось).
        """
        if not entries:
            return
        path = self._archive_path(target)
        try:
            existing = self._read_file(path)
            self._write_file(path, existing + entries)
        except Exception:
            logger.warning(
                "ARIFLAME: не удалось записать архив памяти %s — %s записей вытеснены "
                "только из промпта", path.name, len(entries), exc_info=True,
            )

    def _fit_within_limit(
        self, target: str, working: List[str], protected: List[str]
    ) -> Tuple[List[str], Any]:
        """Уложить ``working`` в лимит хранилища, освобождая место.

        Возвращает (записи, отчёт). Отчёт с ``fits=False`` означает, что
        освобождать больше нечего — так бывает ровно в одном случае: сама
        новая запись длиннее всего хранилища.
        """
        from agent.memory_reclaim import reclaim

        protected_idx = [i for i, e in enumerate(working) if e in set(protected)]
        result = reclaim(
            working,
            self._char_limit(target),
            delimiter=ENTRY_DELIMITER,
            protected=protected_idx,
            normalize=normalize_for_match,
        )
        if result.evicted:
            self._archive(target, result.evicted)
            logger.info(
                "ARIFLAME: память %s переполнена — %s записей вытеснены в %s",
                target, len(result.evicted), self._archive_path(target).name,
            )
        return result.entries, result

    def replace_all(self, target: str, entries: List[str], expected: Optional[List[str]] = None) -> bool:
        """Заменить хранилище целиком (точка входа фоновой консолидации).

        ``expected`` — состояние, которое консолидация видела, когда
        начинала. Если за время прогона память успела измениться (человек
        что-то сказал, агент что-то записал), результат применять НЕЛЬЗЯ:
        он собран из устаревшего списка и затрёт свежую запись. Такое
        сравнение дешевле любых версий и блокировок и ловит ровно тот
        случай, который бывает.
        """
        entries = [e.strip() for e in entries if e and e.strip()]
        if not entries:
            return False
        if len(ENTRY_DELIMITER.join(entries)) > self._char_limit(target):
            return False
        with self._file_lock(self._path_for(target)):
            self._reload_target(target, skip_drift=True)
            if expected is not None and self._entries_for(target) != list(expected):
                logger.info(
                    "ARIFLAME: консолидация %s отброшена — память изменилась во время прогона",
                    target,
                )
                return False
            self._set_entries(target, entries)
            self.save_to_disk(target)
        return True

    def _maybe_consolidate(self, target: str) -> None:
        """Запустить фоновую консолидацию, когда до стены осталось немного.

        Порог 90%, а не 100%, намеренно: консолидация должна успеть до
        того, как вытеснение начнёт выбрасывать записи. Стоит она один
        дешёвый запрос и не задерживает ход.
        """
        if not self._flag("auto_consolidate"):
            return
        limit = self._char_limit(target)
        if limit <= 0 or self._char_count(target) < int(limit * 0.9):
            return
        try:
            from agent.memory_reclaim import consolidate_in_background

            consolidate_in_background(self, target, delimiter=ENTRY_DELIMITER)
        except Exception:
            logger.debug("ARIFLAME: фоновая консолидация не запустилась", exc_info=True)

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # For add (append-only), we skip the drift guard — appending never
            # clobbers existing content, so round-trip mismatches from prior
            # tool-written entries in the same session are harmless.  The drift
            # guard remains active for replace/remove where full-file rewrite
            # would discard un-roundtrippable content (issue #26045).
            self._reload_target(target, skip_drift=True)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            reclaimed = None
            if new_total > limit:
                # ARIFLAME: здесь апстрим отказывал и просил модель
                # «сконсолидировать и повторить в этом же ходу». Механизма за
                # просьбой не было, поэтому просьба не работала: три бокса
                # неделю отвечали «запомнила» и не запоминали. Теперь место
                # освобождается само (сжатие → поглощение дублей → вытеснение
                # старого в архив рядом), и НОВЫЙ факт доезжает до диска
                # всегда, кроме случая «одна запись длиннее всего хранилища».
                if not self._flag("auto_reclaim"):
                    return self._capacity_error(target, content, len(content))
                new_entries, reclaimed = self._fit_within_limit(
                    target, new_entries, [content]
                )
                if not reclaimed.fits:
                    return self._capacity_error(target, content, len(content))

            self._set_entries(target, new_entries)
            self.save_to_disk(target)

        response = self._success_response(target, "Entry added.", entry=content)
        if reclaimed is not None and reclaimed.freed_anything:
            response["memory_was_full"] = reclaimed.note()
        self._maybe_consolidate(target)
        return response

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            hits, how = find_entry_indices(entries, old_text)

            if not hits:
                return self._consolidation_failure(
                    self._no_match_error(target, old_text, "replace")
                )

            if len(hits) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {entries[i] for i in hits}
                if len(unique_texts) > 1:
                    return {
                        "success": False,
                        "error": (
                            f"Multiple entries matched '{old_text}'. Reissue with "
                            f"old_text set to one of the ids below (e.g. old_text=\"m:ab12cd\")."
                        ),
                        "matches": [_preview(entries[i]) for i in hits[:_PREVIEW_ENTRIES]],
                    }
                # All identical -- safe to replace just the first

            idx = hits[0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            reclaimed = None
            if new_total > limit:
                # ARIFLAME: то же, что и в add — освобождаем место сами
                # вместо просьбы к модели сделать это за нас.
                if not self._flag("auto_reclaim"):
                    return self._capacity_error(target, new_content, len(new_content))
                test_entries, reclaimed = self._fit_within_limit(
                    target, test_entries, [new_content]
                )
                if not reclaimed.fits:
                    return self._capacity_error(target, new_content, len(new_content))

            self._set_entries(target, test_entries)
            self.save_to_disk(target)

        response = self._success_response(target, "Entry replaced.", entry=new_content)
        if how and how != "exact":
            response["matched_by"] = how
        if reclaimed is not None and reclaimed.freed_anything:
            response["memory_was_full"] = reclaimed.note()
        self._maybe_consolidate(target)
        return response

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            hits, how = find_entry_indices(entries, old_text)

            if not hits:
                return self._consolidation_failure(
                    self._no_match_error(target, old_text, "remove")
                )

            if len(hits) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {entries[i] for i in hits}
                if len(unique_texts) > 1:
                    return {
                        "success": False,
                        "error": (
                            f"Multiple entries matched '{old_text}'. Reissue with "
                            f"old_text set to one of the ids below (e.g. old_text=\"m:ab12cd\")."
                        ),
                        "matches": [_preview(entries[i]) for i in hits[:_PREVIEW_ENTRIES]],
                    }
                # All identical -- safe to remove just the first

            idx = hits[0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        response = self._success_response(target, "Entry removed.")
        if how and how != "exact":
            response["matched_by"] = how
        return response

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        All operations are validated and applied against the FINAL budget --
        intermediate overflow is irrelevant. This lets the model free space
        (remove/replace) and add new entries in a SINGLE tool call instead of
        the multi-turn consolidate-then-retry dance that re-sends the whole
        conversation context several times.

        Semantics: all-or-nothing. If any op is malformed, doesn't match, or
        the net result would exceed the char limit, NOTHING is written and an
        error is returned describing the first failure plus the live state.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content for injection/exfil BEFORE touching
        # disk -- a single poisoned op rejects the whole batch.
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            # Work on a copy; only commit if the whole batch validates.
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)
            # ARIFLAME: записи, которых касается ЭТОТ батч. Вытеснение не
            # имеет права выбросить то, что модель только что попросила
            # сохранить, — иначе освобождение места съест свой же результат.
            touched: List[str] = []

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue  # idempotent -- skip duplicate, don't fail the batch
                    working.append(content)
                    touched.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    # ARIFLAME: тот же каскад адресации, что и в одиночных
                    # операциях. Батч — основной путь консолидации, и промах
                    # регистром здесь ронял ВЕСЬ набор правок, а не одну.
                    matches, _how = find_entry_indices(working, old_text)
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched '{old_text}' in "
                                    f"{self._path_for(target).name}."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- "
                            f"address one by id instead (old_text=\"m:ab12cd\").",
                        )
                    working[matches[0]] = content
                    touched.append(content)

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches, _how = find_entry_indices(working, old_text)
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched '{old_text}' in "
                                    f"{self._path_for(target).name}."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- "
                            f"address one by id instead (old_text=\"m:ab12cd\").",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            reclaimed = None
            if new_total > limit:
                longest = max((len(e) for e in touched), default=0)
                if not self._flag("auto_reclaim"):
                    return self._capacity_error(target, None, longest)
                working, reclaimed = self._fit_within_limit(target, working, touched)
                if not reclaimed.fits:
                    return self._capacity_error(target, None, longest)

            # Commit.
            self._set_entries(target, working)
            self.save_to_disk(target)

        response = self._success_response(target, f"Applied {len(operations)} operation(s).")
        if reclaimed is not None and reclaimed.freed_anything:
            response["memory_was_full"] = reclaimed.note()
        self._maybe_consolidate(target)
        return response

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            # ARIFLAME: выжимка вместо всей памяти — см. entries_digest.
            "current_entries": entries_digest(self._entries_for(target)),
            "usage": f"{current:,}/{limit:,}",
        })

    # ── ARIFLAME: короткие и адресные ошибки ──────────────────────────

    def _capacity_error(self, target: str, content: Optional[str], size: int) -> Dict[str, Any]:
        """Отказ по месту, который остался возможным ровно в одном случае.

        Освобождение места умеет всё, кроме чуда: запись, которая одна
        длиннее целого хранилища, не поместится никогда. Здесь не нужен ни
        список записей, ни просьба консолидировать — нужна одна цифра и
        одно требование.
        """
        limit = self._char_limit(target)
        current = self._char_count(target)
        return {
            "success": False,
            "done": True,
            "error": (
                f"This entry is {size:,} chars and would exceed the {target} store "
                f"limit of {limit:,} chars even after freeing every other entry. "
                f"Split it into separate short facts (one fact per entry, "
                f"under 120 chars) and retry."
            ),
            "usage": f"{current:,}/{limit:,}",
        }

    def _no_match_error(self, target: str, old_text: str, action: str) -> Dict[str, Any]:
        """Отказ «не нашёл», который говорит ГДЕ искал и что рядом.

        Апстримовский текст не называл хранилище и прикладывал все записи —
        и модель повторяла тот же вызов слово в слово (четыре пары
        идентичных повторов у одной клиентки, две у другой; один повтор
        пришёл через сутки). Здесь три вещи, которых там не было: имя файла,
        перечень ступеней поиска (чтобы было видно, что «попробуй ещё раз
        так же» бессмысленно) и ИДЕНТИФИКАТОРЫ ближайших записей — по ним
        можно попасть точно.
        """
        entries = self._entries_for(target)
        filename = self._path_for(target).name
        current = self._char_count(target)
        limit = self._char_limit(target)
        return {
            "success": False,
            "error": (
                f"No entry matched '{old_text}' in {filename} ({len(entries)} entries). "
                f"Tried exact text, then case/quote/spacing-insensitive, then nearest "
                f"match — repeating this call unchanged will fail the same way. "
                f"Either {action} one of the entries listed below by its id "
                f"(old_text=\"m:ab12cd\"), or, if the fact is new, add it."
            ),
            "closest_entries": self._closest(entries, old_text),
            "usage": f"{current:,}/{limit:,}",
        }

    @staticmethod
    def _closest(entries: List[str], needle: str) -> List[str]:
        """Несколько записей, наиболее похожих на промах, с идентификаторами."""
        if not entries:
            return []
        n = normalize_for_match(needle)
        if not n:
            return [_preview(e) for e in entries[-_PREVIEW_ENTRIES:]]
        ranked = sorted(
            entries,
            key=lambda e: difflib.SequenceMatcher(None, n, normalize_for_match(e)).ratio(),
            reverse=True,
        )
        return [_preview(e) for e in ranked[:_PREVIEW_ENTRIES]]

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None,
                          entry: Optional[str] = None) -> Dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures,
        # not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        # ARIFLAME: идентификатор только что записанного факта. Он стоит
        # десять символов и снимает целый класс промахов: следующая правка
        # этой записи адресуется точно, а не пересказом её начала.
        if entry:
            resp["id"] = entry_marker(entry)
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"{MEMORY_BLOCK_HEADERS['user']} [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"{MEMORY_BLOCK_HEADERS['memory']} [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        # ARIFLAME: закрывающий разделитель. Блок был открыт с одного конца:
        # найти, где он КОНЧАЕТСЯ, было нельзя — за ним сразу шли другие
        # строки промпта, а записи бывают многострочными. Сборщику дельты
        # (agent/memory_delta.py) нужно понимать, что модель уже видит,
        # причём разбирая промпт, вытащенный из базы, а не собранный сейчас.
        # Сорок семь символов на блок; для модели граница блока тоже честнее.
        return f"{separator}\n{header}\n{separator}\n{content}\n{separator}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _detect_external_drift(self, target: str) -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def load_on_disk_store() -> "MemoryStore":
    """Build a fresh on-disk :class:`MemoryStore`, honoring configured char limits.

    Use this from any context that has no live agent (the messaging gateway, the
    Desktop GUI, the bare CLI ``/memory`` handler) but still needs to read or
    apply approved memory writes. Mirrors how the live agent constructs its store
    in ``agent/agent_init.py`` — including the user's ``memory.memory_char_limit``
    / ``memory.user_char_limit`` overrides — so an approval applied without a live
    agent enforces the SAME caps as one applied with one.

    Falls back to the built-in defaults if config can't be loaded, so this can
    never raise on a missing/unreadable config.
    """
    memory_char_limit = 2200
    user_char_limit = 1375
    try:
        from hermes_cli.config import load_config

        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        memory_char_limit = int(mem_cfg.get("memory_char_limit", memory_char_limit))
        user_char_limit = int(mem_cfg.get("user_char_limit", user_char_limit))
    except Exception:
        pass  # config optional — fall back to defaults rather than break /memory

    store = MemoryStore(
        memory_char_limit=memory_char_limit,
        user_char_limit=user_char_limit,
    )
    store.load_from_disk()
    return store


def _apply_write_gate(action: str, target: str, content: Optional[str],
                      old_text: Optional[str]) -> Optional[str]:
    """Evaluate the memory write gate. Returns a JSON tool-result string when
    the write should NOT proceed normally (blocked or staged), or None when the
    caller should perform the real write.

    Only the mutating actions (add/replace/remove) are gated.
    """
    if action not in {"add", "replace", "remove"}:
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        # If the gate module can't load, fail open (current behaviour) rather
        # than blocking all memory writes.
        return None

    # Build a small inline summary/detail for the foreground approval prompt.
    label = "user profile" if target == "user" else "memory"
    if action == "add":
        summary = f"add to {label}"
        detail = content or ""
    elif action == "replace":
        summary = f"replace in {label}"
        detail = f"old: {old_text}\nnew: {content}"
    else:  # remove
        summary = f"remove from {label}"
        detail = old_text or ""

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    # stage
    payload = {
        "action": action,
        "target": target,
        "content": content,
        "old_text": old_text,
    }
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _apply_batch_write_gate(target: str, operations: List[Dict[str, Any]]) -> Optional[str]:
    """Evaluate the write gate for a batch of memory operations.

    Returns a JSON tool-result string when the batch should NOT proceed
    (blocked or staged), or None when the caller should perform the real
    batch write. The whole batch is gated as a single unit.
    """
    try:
        from tools import write_approval as wa
    except Exception:
        return None

    label = "user profile" if target == "user" else "memory"
    summary = f"apply {len(operations)} op(s) to {label}"
    detail_lines = []
    for op in operations:
        op = op or {}
        act = op.get("action", "?")
        if act == "remove":
            detail_lines.append(f"- remove: {op.get('old_text', '')}")
        elif act == "replace":
            detail_lines.append(f"- replace: {op.get('old_text', '')} -> {op.get('content', '')}")
        else:
            detail_lines.append(f"- {act}: {op.get('content', '')}")
    detail = "\n".join(detail_lines)

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    payload = {"action": "batch", "target": target, "operations": operations}
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store.live_entries(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- the id of the entry (e.g. "
                f"old_text=\"m:ab12cd\") or a short unique substring of it. None was "
                f"provided. Reissue the {action} with old_text set to one of the "
                f"entries below."
            ),
            # ARIFLAME: выжимка, а не весь список — в апстриме этот путь
            # тоже возвращал всю память целиком.
            "current_entries": entries_digest(entries),
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def _resolve_target_for_edit(
    store: "MemoryStore", target: str, old_text: str, target_given: bool
) -> Tuple[str, str]:
    """Выбрать хранилище для правки по тому, где запись НА САМОМ ДЕЛЕ лежит.

    Хранилищ два, и модель регулярно ошибается адресом: границу между «кто
    человек» и «что за дело» она проводит не там, где провели её мы, а поле
    ``target`` вдобавок молча дефолтилось. Ошибиться здесь бесплатно
    невозможно: правка не находится, ответ приходит длиной в двадцать тысяч
    символов, и модель повторяет тот же вызов.

    Правило простое и не спорит с явным указанием без причины: если в
    названном хранилище запись нашлась — работаем там. Если не нашлась, а в
    соседнем нашлась однозначно — работаем там и пишем в лог. Если не
    нашлась нигде — оставляем названное, чтобы ошибка говорила про то
    хранилище, которое модель имела в виду.
    """
    other = "user" if target == "memory" else "memory"
    here, _ = find_entry_indices(store.live_entries(target), old_text)
    if here:
        return target, ""
    there, how = find_entry_indices(store.live_entries(other), old_text)
    if len(there) == 1 or (there and len({store.live_entries(other)[i] for i in there}) == 1):
        return other, (
            f"память: правка '{old_text[:40]}' адресована в {target}, "
            f"а запись лежит в {other} (совпало: {how}"
            f"{'' if target_given else ', target не был указан'}) — правим там"
        )
    return target, ""


def memory_tool(
    action: str = None,
    target: Optional[str] = None,
    content: str = None,
    old_text: str = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Two shapes:
      - Single op: action + (content / old_text).
      - Batch:     operations=[{action, content?, old_text?}, ...] applied
                   atomically against the final char budget in ONE call.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    # Some strict providers fill optional schema fields with JSON null rather
    # than omitting them.  Treat ``target: null`` as omitted so memory writes
    # still use the documented default store instead of failing validation.
    # ARIFLAME: но «не указано» и «указано memory» — это РАЗНЫЕ вещи, и
    # апстрим их путал: молчаливый дефолт означал, что правка искалась в
    # MEMORY.md, когда запись лежит в USER.md (6 промахов из 14 у одной
    # клиентки, в двух вызовах поле вообще не передавалось). Для правок
    # хранилище теперь ВЫЧИСЛЯЕТСЯ по тому, где запись действительно лежит.
    target_given = target is not None
    if target is None:
        target = "memory"

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    corrected_from = ""
    if action in {"replace", "remove"} and old_text:
        resolved, note = _resolve_target_for_edit(store, target, old_text, target_given)
        if note:
            logger.info("ARIFLAME: %s", note)
            corrected_from = target
        target = resolved

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        gate_result = _apply_batch_write_gate(target, operations)
        if gate_result is not None:
            return gate_result
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # The client/model omitted old_text. Replace is inherently targeted
            # -- we can't guess which entry. Return the current inventory plus a
            # retry instruction so the model can reissue with old_text set,
            # instead of hitting a dead-end error. (issues #43412, #49466)
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    # Approval gate: when on, stages the write (background/gateway) or prompts
    # inline (interactive CLI); when off (default) passes straight through.
    gate_result = _apply_write_gate(action, target, content, old_text)
    if gate_result is not None:
        return gate_result

    if action == "add":
        result = store.add(target, content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    # ARIFLAME: если хранилище пришлось поправить — сказать об этом модели, а
    # не молча сделать по-своему. Иначе она в следующий раз ошибётся так же.
    if corrected_from and isinstance(result, dict) and result.get("success"):
        result["target_corrected"] = (
            f"the entry lives in '{target}', not '{corrected_from}' — "
            f"address it there next time"
        )

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


def apply_memory_pending(payload: Dict[str, Any], store: "MemoryStore") -> Dict[str, Any]:
    """Replay a staged memory write directly against the store, bypassing the
    write gate. Called by the /memory approve handler.

    Returns the store's result dict.
    """
    action = payload.get("action")
    target = payload.get("target", "memory")
    content = payload.get("content") or ""
    old_text = payload.get("old_text") or ""
    if action == "batch":
        return store.apply_batch(target, payload.get("operations") or [])
    if action == "add":
        return store.add(target, content)
    if action == "replace":
        return store.replace(target, old_text, content)
    if action == "remove":
        return store.remove(target, old_text)
    return {"success": False, "error": f"Unknown staged action '{action}'."}
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: nothing to do — the store frees room by itself (merges duplicates, then "
        "moves its oldest entries to an archive file). Your write lands. When that happens "
        "the response carries 'memory_was_full'; mention it to the user only if the archived "
        "facts mattered.\n\n"
        "ADDRESSING AN ENTRY: every successful write returns an 'id' like m:ab12cd, and error "
        "responses list ids next to entries. Pass that id as old_text to replace/remove that "
        "exact entry. A substring still works too (case, quotes and spacing are ignored). If "
        "an entry is not found, the response says which file was searched — do NOT reissue the "
        "same call unchanged; use an id from the list or add the fact as a new entry.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons). For replace/remove the store "
        "is resolved by where the entry actually is, so a wrong guess is corrected, not failed.\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform (single-op shape). Omit when using 'operations'."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        # ARIFLAME: НЕ подставляем "memory" за модель — отсутствие поля
        # означает «не знаю, где», и это решается поиском, а не дефолтом.
        target=args.get("target"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        operations=args.get("operations"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




