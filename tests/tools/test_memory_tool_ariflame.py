"""ARIFLAME: адрес записи в памяти и цена неудачного вызова.

Каждый тест здесь соответствует промаху, СЛУЧИВШЕМУСЯ на живом боксе, а не
придуманному сценарию: разбор 20.08.2026 (docs/incidents/2026-08-20-memory.md)
разложил 14 неудачных вызовов памяти одной клиентки на четыре класса, и три
из них лечатся здесь.
"""

import json

import pytest

from tools.memory_tool import (
    MemoryStore,
    entries_digest,
    entry_marker,
    find_entry_indices,
    memory_tool,
    normalize_for_match,
    parse_marker,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    # Лимиты как на живых боксах: 60000/30000. Тесты вытеснения живут
    # отдельно, здесь оно только мешало бы читать проверки адресации.
    s = MemoryStore(memory_char_limit=60000, user_char_limit=30000,
                    auto_reclaim=True, auto_consolidate=False)
    s.load_from_disk()
    return s


class TestMarker:
    def test_marker_is_stable_and_short(self):
        m = entry_marker("Счета выставляем от ИП, НДС не облагается")
        assert m == entry_marker("Счета выставляем от ИП, НДС не облагается")
        assert m.startswith("m:") and len(m) == 8

    def test_marker_survives_case_and_spacing(self):
        # Ровно тот случай, из-за которого разошлись два сообщения: модель
        # процитировала фразу с другой первой буквой.
        assert entry_marker("Подпись под визуалами") == entry_marker("подпись   под визуалами")

    def test_marker_changes_with_meaning(self):
        assert entry_marker("цена 5000") != entry_marker("цена 6000")

    def test_parse_marker_accepts_both_spellings(self):
        assert parse_marker("m:ab12cd") == "m:ab12cd"
        assert parse_marker("[m:AB12CD]") == "m:ab12cd"
        assert parse_marker("подпись под визуалами") is None


class TestNormalize:
    def test_quotes_dashes_yo_and_spaces_are_the_same_text(self):
        a = normalize_for_match("«Студия Владислав» — всё")
        b = normalize_for_match('"студия  владислав" - все')
        assert a == b


class TestFindCascade:
    ENTRIES = [
        "Подпись под визуалами — «Студия Владислав»",
        "Счета выставляем от ИП, НДС не облагается",
        "Съёмки по вторникам после 15:00",
    ]

    def test_exact_substring_still_wins(self):
        hits, how = find_entry_indices(self.ENTRIES, "Счета выставляем")
        assert hits == [1] and how == "exact"

    def test_case_mismatch_matches(self):
        # «П» против «п» — потерянная правка на живом боксе.
        hits, how = find_entry_indices(self.ENTRIES, "подпись под визуалами")
        assert hits == [0] and how == "normalized"

    def test_quote_style_mismatch_matches(self):
        hits, how = find_entry_indices(self.ENTRIES, '"Студия Владислав"')
        assert hits == [0] and how == "normalized"

    def test_marker_matches_exactly_one(self):
        marker = entry_marker(self.ENTRIES[2])
        hits, how = find_entry_indices(self.ENTRIES, marker)
        assert hits == [2] and how == "id"

    def test_paraphrase_close_enough_matches_fuzzily(self):
        hits, how = find_entry_indices(self.ENTRIES, "Счета выставляем от ИП, НДС не облагаются")
        assert hits == [1] and how == "fuzzy"

    def test_short_needle_is_not_guessed(self):
        # На трёх символах «похоже» означает «случайно» — лучше отказ.
        assert find_entry_indices(self.ENTRIES, "ИП!") == ([], "")

    def test_ambiguous_paraphrase_refuses_rather_than_picks(self):
        entries = ["сервер A под nginx", "сервер B под nginx"]
        hits, how = find_entry_indices(entries, "сервер под nginx, который")
        assert hits == [] or how == "normalized"


class TestErrorCost:
    def test_no_match_error_names_the_file_and_stays_small(self, store):
        for i in range(60):
            store.add("memory", f"запись номер {i} " + "и" * 250)
        result = store.remove("memory", "того чего там нет никогда")
        assert result["success"] is False
        assert "MEMORY.md" in result["error"]
        assert "60 entries" in result["error"]
        # Апстрим на этом месте отдавал ~20 000 символов.
        assert len(json.dumps(result, ensure_ascii=False)) < 1500
        assert all(e.startswith("[m:") for e in result["closest_entries"])

    def test_no_match_error_tells_the_model_not_to_repeat(self, store):
        store.add("memory", "запись")
        result = store.remove("memory", "нет такого")
        assert "repeating this call unchanged" in result["error"]

    def test_digest_passes_small_lists_through_untouched(self):
        assert entries_digest(["fact A", "fact B"]) == ["fact A", "fact B"]

    def test_digest_shrinks_a_real_memory_file(self):
        entries = [f"факт {i} " + "ю" * 260 for i in range(130)]
        digest = entries_digest(entries)
        assert len(digest) <= 7
        assert "ещё 124" in digest[0]
        assert len("\n".join(digest)) < 1200


class TestTargetResolution:
    def test_edit_finds_the_entry_in_the_other_store(self, store):
        store.add("user", "Пишем без восклицательных знаков")
        # Модель адресует правку в memory (или вообще не указывает target) —
        # запись лежит в user. Апстрим отвечал «не нашёл» и прикладывал
        # чужие 133 записи; 6 промахов из 14 у одной клиентки — это он.
        out = json.loads(memory_tool(
            action="replace",
            old_text="без восклицательных",
            content="Пишем без восклицательных знаков и без капса",
            store=store,
        ))
        assert out["success"] is True
        assert out["target"] == "user"
        assert "target_corrected" in out
        assert store.user_entries == ["Пишем без восклицательных знаков и без капса"]

    def test_explicit_target_is_respected_when_the_entry_is_there(self, store):
        store.add("memory", "одинаковый текст")
        store.add("user", "одинаковый текст")
        out = json.loads(memory_tool(
            action="remove", target="user", old_text="одинаковый текст", store=store))
        assert out["success"] is True
        assert store.memory_entries == ["одинаковый текст"]
        assert store.user_entries == []

    def test_missing_everywhere_reports_the_named_store(self, store):
        store.add("user", "что-то своё")
        out = json.loads(memory_tool(
            action="remove", target="memory", old_text="нет нигде такого текста", store=store))
        assert out["success"] is False
        assert "MEMORY.md" in out["error"]


class TestSuccessCarriesId:
    def test_add_returns_the_id_of_what_it_wrote(self, store):
        out = store.add("memory", "Съёмки по вторникам после 15:00")
        assert out["id"] == entry_marker("Съёмки по вторникам после 15:00")

    def test_the_id_can_be_used_to_replace_that_entry(self, store):
        ident = store.add("memory", "Съёмки по вторникам")["id"]
        out = store.replace("memory", ident, "Съёмки по средам")
        assert out["success"] is True
        assert out["matched_by"] == "id"
        assert store.memory_entries == ["Съёмки по средам"]
