"""ARIFLAME: вытеснение и консолидация памяти — то, чего в апстриме нет.

Апстрим предлагает модели «сконсолидируй и повтори в этом же ходу», а сам
не умеет освобождать место ничем. Механизма за словом «консолидация» не
стоит ни одной строки: это ТЕКСТ ПРОСЬБЫ в теле ошибки. Дальше происходит
следующее — модель четыре раза подряд не попадает в лимит, рантайм отвечает
«Memory consolidation failed 4 times this turn… The fact can be saved in a
later turn», и на этом всё. Обещание пустое: в следующем ходу свободного
места ровно столько же, сколько было. На трёх живых боксах память таким
образом умерла совсем — у одной клиентки в USER.md оставалось 12 свободных
символов, и она неделю разговаривала с агентом, который на каждое «запомни»
отвечал «запомнила» и не запоминал.

Здесь три уровня освобождения места, от бесплатного к дорогому:

1. **Сжатие формы.** Кратные пробелы и пустые строки. Ничего не теряется,
   стоит ноль.
2. **Поглощение.** Запись, целиком содержащаяся в другой, — это дубль,
   оставшийся от «уточню и добавлю ещё раз». Теряется только повтор.
3. **Вытеснение в архив.** Самые старые записи уезжают в отдельный файл
   рядом. Это НЕ удаление: файл лежит на диске, его можно прочитать и
   вернуть. Из промпта запись уходит, с диска — нет.

Порядок именно такой, и вытеснение — последнее. Но даже последнее лучше
апстримового поведения: сегодня выбор стоит не между «выкинуть старое» и
«сохранить всё», а между «выкинуть старое» и «молча потерять НОВОЕ, сказав
человеку, что запомнили». Второе мы уже год делаем, и это худший из двух.

Четвёртый уровень — настоящая консолидация дешёвой моделью — живёт
отдельно (:func:`consolidate_in_background`) и работает ВНЕ диалога: в
диалоге модель после четырёх неудач сдаётся, и правильно делает — её
задача отвечать человеку, а не воевать с лимитом. Запускается заранее, на
90% заполнения, чтобы к моменту стены места уже хватало.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Что в записи считаем ценным настолько, чтобы вытеснять её последней.
# Цифры (цены, сроки, реквизиты), ссылки, аккаунты и — главное — запреты.
# Правило-поправка вида «не пиши так официально» действует навсегда, и
# потерять его больнее, чем потерять описание закрытого проекта.
_VALUABLE_RE = re.compile(
    r"(https?://|@[\w.\-]+|\d\d|[₽$€]"
    r"|\b(?:не|нет|нельзя|никогда|запрещ\w*|только|обязательно|всегда"
    r"|never|always|don't|do not|must)\b)",
    re.IGNORECASE,
)

# То, что консолидация обязана сохранить дословно. Если хоть один такой
# токен пропал из результата — результат не применяется целиком. Это
# единственная проверка, которая ловит «модель пересказала своими словами и
# потеряла цену».
_MUST_KEEP_RE = re.compile(r"(https?://\S+|[\w.\-]+@[\w.\-]+|@[\w.\-]{2,}|\d{2,})")

_MAX_ENTRY_SQUEEZE_NEWLINES = re.compile(r"\n{3,}")
_MAX_ENTRY_SQUEEZE_SPACES = re.compile(r"[ \t]{2,}")

# Одновременно бежит не больше одной консолидации на процесс: она ходит в
# сеть и пишет в файл под локом, а второй такой же прогон не ускорит
# ничего, зато удвоит счёт.
_consolidation_lock = threading.Lock()
_consolidation_running: Dict[str, bool] = {}


@dataclass
class ReclaimResult:
    """Что получилось освободить и какой ценой."""

    entries: List[str]
    fits: bool
    evicted: List[str] = field(default_factory=list)
    squeezed: int = 0
    absorbed: int = 0

    @property
    def freed_anything(self) -> bool:
        return bool(self.evicted) or self.squeezed > 0 or self.absorbed > 0

    def note(self) -> str:
        """Одна строка для модели: что произошло с памятью, кроме её записи."""
        parts = []
        if self.absorbed:
            parts.append(f"{self.absorbed} duplicate entries merged")
        if self.squeezed:
            parts.append(f"{self.squeezed} entries reflowed")
        if self.evicted:
            parts.append(
                f"{len(self.evicted)} oldest entries moved to the archive file "
                f"(kept on disk, no longer in your prompt)"
            )
        return "; ".join(parts)


def _squeeze(entry: str) -> str:
    """Убрать кратные пробелы и пустые строки, не трогая смысл."""
    out = _MAX_ENTRY_SQUEEZE_SPACES.sub(" ", entry)
    out = _MAX_ENTRY_SQUEEZE_NEWLINES.sub("\n\n", out)
    return out.strip()


def _total(entries: Sequence[str], delimiter: str) -> int:
    return len(delimiter.join(entries)) if entries else 0


def is_valuable(entry: str) -> bool:
    """Есть ли в записи то, что нельзя восстановить одним вопросом."""
    return bool(_VALUABLE_RE.search(entry or ""))


def reclaim(
    entries: Sequence[str],
    limit: int,
    *,
    delimiter: str,
    protected: Optional[Sequence[int]] = None,
    normalize: Callable[[str], str],
) -> ReclaimResult:
    """Освободить место под ``limit``, начиная с самого дешёвого способа.

    ``protected`` — индексы записей, которых операция касается прямо сейчас
    (та самая новая запись). Их нельзя ни поглотить, ни вытеснить: смысл
    всей работы в том, чтобы НОВЫЙ факт дошёл до диска.

    ``normalize`` передаётся снаружи (``memory_tool.normalize_for_match``),
    чтобы поглощение считало одинаковыми ровно то же, что считает
    одинаковым поиск, — иначе тул нашёл бы запись, которую вытеснение
    только что признало дублем и удалило.
    """
    working = [e for e in entries]
    protected_set = set(protected or ())
    protected_texts = {working[i] for i in protected_set if 0 <= i < len(working)}

    if _total(working, delimiter) <= limit:
        return ReclaimResult(entries=working, fits=True)

    # 1. Сжатие формы.
    squeezed = 0
    for i, e in enumerate(working):
        s = _squeeze(e)
        if s != e and s:
            working[i] = s
            squeezed += 1
    if _total(working, delimiter) <= limit:
        return ReclaimResult(entries=working, fits=True, squeezed=squeezed)

    # 2. Поглощение: запись, целиком лежащая внутри другой, — это повтор.
    norm = [normalize(e) for e in working]
    absorbed_idx = set()
    for i in range(len(working)):
        if working[i] in protected_texts or not norm[i]:
            continue
        for j in range(len(working)):
            if i == j or j in absorbed_idx or not norm[j]:
                continue
            if len(norm[i]) < len(norm[j]) and norm[i] in norm[j]:
                absorbed_idx.add(i)
                break
    if absorbed_idx:
        working = [e for i, e in enumerate(working) if i not in absorbed_idx]
    if _total(working, delimiter) <= limit:
        return ReclaimResult(
            entries=working, fits=True, squeezed=squeezed, absorbed=len(absorbed_idx)
        )

    # 3. Вытеснение. Сначала старые записи без цифр, ссылок и запретов;
    # только если этого не хватило — старые ценные. Порядок в файле и есть
    # возраст: тул всегда дописывает в конец.
    evicted: List[str] = []
    order = [i for i, e in enumerate(working) if e not in protected_texts and not is_valuable(e)]
    order += [i for i, e in enumerate(working) if e not in protected_texts and is_valuable(e)]
    doomed = set()
    for idx in order:
        if _total([e for i, e in enumerate(working) if i not in doomed], delimiter) <= limit:
            break
        doomed.add(idx)
    if doomed:
        evicted = [e for i, e in enumerate(working) if i in doomed]
        working = [e for i, e in enumerate(working) if i not in doomed]

    return ReclaimResult(
        entries=working,
        fits=_total(working, delimiter) <= limit,
        evicted=evicted,
        squeezed=squeezed,
        absorbed=len(absorbed_idx),
    )


# ---------------------------------------------------------------------------
# Настоящая консолидация: отдельный дешёвый прогон вне диалога.
# ---------------------------------------------------------------------------

_CONSOLIDATION_INSTRUCTIONS = (
    "You are compacting a personal-assistant memory file. It is a list of "
    "facts about one person and their work.\n\n"
    "Rewrite the SAME facts in fewer characters. Rules:\n"
    "- Keep every fact. You may merge two entries that are about the same "
    "thing, but you may not drop information.\n"
    "- Keep the original language of each entry. Do not translate.\n"
    "- Keep every number, price, date, URL, handle and proper name EXACTLY "
    "as written.\n"
    "- Keep every prohibition and correction (\"never\", \"не\", \"нельзя\") — "
    "these are standing rules, they are the most valuable lines here.\n"
    "- Drop filler: the person's own name at the start of a line, "
    "\"prefers/wants/requires\", polite framing. Write facts as short "
    "imperative statements.\n"
    "- One fact per entry, at most 120 characters per entry.\n\n"
    "Return ONLY a JSON array of strings — the new entries, in the same "
    "order as the input. No prose, no code fence, no keys."
)


def _must_keep_tokens(text: str) -> set:
    return set(_MUST_KEEP_RE.findall(text or ""))


def validate_consolidation(
    original: Sequence[str], produced: Any, limit: int, *, delimiter: str
) -> Tuple[Optional[List[str]], str]:
    """Принять результат консолидации или отвергнуть его целиком.

    Ни одна проверка здесь не про красоту: каждая ловит способ, которым
    дешёвая модель молча теряет данные. Полумер нет — либо результат
    применяется весь, либо не применяется вовсе: применить «частично
    сконсолидированный» файл значит получить память, про которую никто не
    знает, что в ней осталось.
    """
    if not isinstance(produced, list) or not produced:
        return None, "не список записей"
    entries = [str(e).strip() for e in produced if str(e or "").strip()]
    if not entries:
        return None, "пустой результат"
    if any(len(e) > limit for e in entries):
        return None, "запись длиннее всего хранилища"

    # Консолидация имеет право укорачивать, но не удалять. Если записей
    # стало меньше почти вдвое — это уже не «слил похожие», а «выбросил
    # лишнее по своему усмотрению».
    if len(entries) < max(1, int(len(original) * 0.6)):
        return None, f"записей стало {len(entries)} из {len(original)}"

    new_total = _total(entries, delimiter)
    old_total = _total(original, delimiter)
    if new_total >= old_total:
        return None, "короче не стало"

    lost = _must_keep_tokens(delimiter.join(original)) - _must_keep_tokens(
        delimiter.join(entries)
    )
    if lost:
        sample = ", ".join(sorted(lost)[:5])
        return None, f"потеряны точные значения: {sample}"

    return entries, ""


def _run_consolidation(store: Any, target: str, delimiter: str) -> None:
    """Один прогон консолидации. Вызывается ТОЛЬКО из фонового потока."""
    try:
        from agent.oneshot import run_oneshot
    except Exception:
        logger.debug("ARIFLAME: консолидация памяти недоступна (нет oneshot)", exc_info=True)
        return

    entries = list(store.live_entries(target))
    if len(entries) < 4:
        return
    limit = store.char_limit(target)
    payload = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(entries))

    text = ""
    # Задача берётся из конфига по цепочке: своя настройка, иначе настройка
    # компакции (на боксах она всегда задана и всегда дешёвая). Ставить сюда
    # основную модель нельзя — это тот же счёт, что и разговор.
    for task in ("memory_consolidation", "compression", "title_generation"):
        try:
            text = run_oneshot(
                instructions=_CONSOLIDATION_INSTRUCTIONS,
                user_input=payload,
                task=task,
                max_tokens=8000,
                temperature=0.2,
                timeout=120.0,
            )
            if text:
                break
        except Exception as exc:
            logger.debug("ARIFLAME: консолидация через %s не прошла: %s", task, exc)
    if not text:
        return

    import json

    try:
        produced = json.loads(text)
    except Exception:
        logger.warning("ARIFLAME: консолидация памяти вернула не JSON — результат отброшен")
        return

    new_entries, why = validate_consolidation(entries, produced, limit, delimiter=delimiter)
    if new_entries is None:
        logger.warning("ARIFLAME: консолидация памяти отвергнута (%s)", why)
        return

    applied = store.replace_all(target, new_entries, expected=entries)
    if applied:
        logger.info(
            "ARIFLAME: память %s сконсолидирована: %s → %s символов, %s → %s записей",
            target,
            _total(entries, delimiter),
            _total(new_entries, delimiter),
            len(entries),
            len(new_entries),
        )


def consolidate_in_background(store: Any, target: str, *, delimiter: str) -> bool:
    """Запустить консолидацию в фоне, если она ещё не бежит.

    Фоном — потому что в ходе разговора ждать сеть нельзя: человек смотрит
    на «печатает…». Ход при этом уже завершился успешно (место освободило
    вытеснение), а консолидация готовит место к следующему разу.
    """
    with _consolidation_lock:
        if _consolidation_running.get(target):
            return False
        _consolidation_running[target] = True

    def _worker() -> None:
        try:
            _run_consolidation(store, target, delimiter)
        except Exception:
            logger.warning("ARIFLAME: фоновая консолидация памяти упала", exc_info=True)
        finally:
            with _consolidation_lock:
                _consolidation_running[target] = False

    threading.Thread(
        target=_worker, name=f"ariflame-memory-consolidate-{target}", daemon=True
    ).start()
    return True
