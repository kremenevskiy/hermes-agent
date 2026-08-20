"""ARIFLAME: вытеснение памяти — то, чего в апстриме не было ни строки.

Проверяем не «функция работает», а три обещания, которых апстрим не давал:
новый факт доезжает до диска ВСЕГДА (кроме записи длиннее хранилища),
вытесненное лежит на диске и его можно прочитать, и результат консолидации
принимается только целиком и только если ничего не потеряно.
"""

import json

import pytest

from agent.memory_reclaim import (
    ReclaimResult,
    is_valuable,
    reclaim,
    validate_consolidation,
)
from tools.memory_tool import ENTRY_DELIMITER, MemoryStore, normalize_for_match

DELIM = ENTRY_DELIMITER


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=500, user_char_limit=300,
                    auto_reclaim=True, auto_consolidate=False)
    s.load_from_disk()
    return s


class TestReclaimLadder:
    def test_nothing_happens_when_it_already_fits(self):
        r = reclaim(["a", "b"], 100, delimiter=DELIM, normalize=normalize_for_match)
        assert r.fits and not r.freed_anything

    def test_squeezing_form_is_tried_before_dropping_anything(self):
        # Одна запись с растянутой формой: сжатие пробелов освобождает
        # достаточно, и ни одна запись не должна пострадать.
        entries = ["правило" + " " * 60 + "первое", "правило второе"]
        r = reclaim(entries, 40, delimiter=DELIM, normalize=normalize_for_match)
        assert r.fits
        assert r.squeezed == 1
        assert not r.evicted
        assert len(r.entries) == 2

    def test_entry_contained_in_another_is_absorbed(self):
        entries = ["съёмки по вторникам", "съёмки по вторникам после 15:00 в студии"]
        r = reclaim(entries, 45, delimiter=DELIM, normalize=normalize_for_match)
        assert r.fits
        assert r.absorbed == 1
        assert r.entries == ["съёмки по вторникам после 15:00 в студии"]

    def test_eviction_takes_the_oldest_cheapest_first(self):
        entries = [
            "старая болтовня без ничего важного",   # старая и дешёвая
            "цена съёмки 15000 рублей",             # цифры — ценная
            "новая болтовня тоже без важного",
        ]
        r = reclaim(entries, 60, delimiter=DELIM, normalize=normalize_for_match,
                    protected=[2])
        assert r.fits
        assert "цена съёмки 15000 рублей" in r.entries
        assert "новая болтовня тоже без важного" in r.entries
        assert r.evicted == ["старая болтовня без ничего важного"]

    def test_protected_entry_is_never_evicted(self):
        entries = ["a" * 100, "b" * 100, "новая запись"]
        r = reclaim(entries, 30, delimiter=DELIM, normalize=normalize_for_match,
                    protected=[2])
        assert "новая запись" in r.entries

    def test_valuable_detection_covers_prohibitions_and_numbers(self):
        assert is_valuable("никогда не ставим восклицательные знаки")
        assert is_valuable("счёт 40817810099910004312")
        assert is_valuable("https://example.com/brief")
        assert not is_valuable("делали пост про осень, зашло нормально")


class TestStoreNeverLosesTheNewFact:
    def test_add_at_capacity_writes_and_archives(self, store):
        # Лимит фикстуры 500. Две записи по 200 занимают 403, третья на 150
        # переполняет — и именно она обязана уцелеть.
        store.add("memory", "древняя запись " + "я" * 185)
        store.add("memory", "ещё одна древняя " + "ю" * 183)
        out = store.add("memory", "правило: счета от ИП, НДС не облагается. " + "-" * 110)
        assert out["success"] is True
        assert any(e.startswith("правило: счета от ИП") for e in store.memory_entries)
        # Вытесненное лежит рядом на диске — это не удаление.
        archive = store._archive_path("memory")
        assert archive.exists()
        assert "древняя запись" in archive.read_text(encoding="utf-8")

    def test_response_says_out_loud_that_memory_was_full(self, store):
        store.add("memory", "x" * 240)
        store.add("memory", "y" * 240)
        out = store.add("memory", "новый факт " + "z" * 100)
        assert "archive" in out["memory_was_full"]

    def test_batch_at_capacity_also_lands(self, store):
        store.add("memory", "x" * 240)
        store.add("memory", "y" * 240)
        out = store.apply_batch("memory", [{"action": "add", "content": "z" * 200}])
        assert out["success"] is True
        assert ("z" * 200) in store.memory_entries

    def test_single_entry_longer_than_the_store_is_the_only_refusal(self, store):
        out = store.add("memory", "q" * 600)
        assert out["success"] is False
        assert "limit" in out["error"].lower()
        assert out["done"] is True
        assert len(json.dumps(out)) < 700  # без дампа памяти

    def test_reclaim_can_be_switched_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore(memory_char_limit=500, user_char_limit=300,
                        auto_reclaim=False, auto_consolidate=False)
        s.load_from_disk()
        s.add("memory", "x" * 490)
        out = s.add("memory", "новый факт, для которого нет места")
        assert out["success"] is False
        # Даже отказ теперь короткий: всей памяти в теле ошибки больше нет.
        assert "current_entries" not in out


class TestConsolidationValidation:
    ORIGINAL = [
        "съёмка стоит 15000 рублей",
        "никогда не ставим восклицательные знаки",
        "подпись под визуалами — Студия Владислав",
        "брифы присылать на brief@example.com",
    ]

    def test_shortened_but_complete_result_is_accepted(self):
        produced = [
            "съёмка 15000 руб",
            "не ставим восклицательные знаки",
            "подпись: Студия Владислав",
            "брифы: brief@example.com",
        ]
        out, why = validate_consolidation(self.ORIGINAL, produced, 10000, delimiter=DELIM)
        assert out == produced and why == ""

    def test_lost_price_rejects_the_whole_result(self):
        produced = [
            "съёмка стоит недорого",
            "не ставим восклицательные знаки",
            "подпись: Студия Владислав",
            "брифы: brief@example.com",
        ]
        out, why = validate_consolidation(self.ORIGINAL, produced, 10000, delimiter=DELIM)
        assert out is None and "потеряны точные значения" in why

    def test_dropping_half_the_entries_rejects_the_result(self):
        produced = ["съёмка 15000 руб, brief@example.com"]
        out, why = validate_consolidation(self.ORIGINAL, produced, 10000, delimiter=DELIM)
        assert out is None and "записей стало" in why

    def test_result_that_is_not_shorter_is_rejected(self):
        produced = [e + " (уточнение)" for e in self.ORIGINAL]
        out, why = validate_consolidation(self.ORIGINAL, produced, 10000, delimiter=DELIM)
        assert out is None and why == "короче не стало"

    def test_garbage_shape_is_rejected(self):
        assert validate_consolidation(self.ORIGINAL, "не список", 10000, delimiter=DELIM)[0] is None
        assert validate_consolidation(self.ORIGINAL, [], 10000, delimiter=DELIM)[0] is None


class TestConsolidationApply:
    def test_result_is_dropped_when_memory_moved_under_it(self, store):
        store.add("memory", "первая")
        store.add("memory", "вторая")
        stale_view = ["первая"]  # консолидация видела память ДО второй записи
        assert store.replace_all("memory", ["слитая"], expected=stale_view) is False
        assert store.memory_entries == ["первая", "вторая"]

    def test_result_is_applied_when_nothing_moved(self, store):
        store.add("memory", "первая")
        store.add("memory", "вторая")
        assert store.replace_all("memory", ["первая и вторая"],
                                 expected=["первая", "вторая"]) is True
        assert store.memory_entries == ["первая и вторая"]

    def test_oversized_result_is_refused(self, store):
        store.add("memory", "первая")
        assert store.replace_all("memory", ["z" * 900], expected=["первая"]) is False
