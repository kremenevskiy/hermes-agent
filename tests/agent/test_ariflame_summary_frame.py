"""ARIFLAME: компакция больше не выбрасывает действующее задание.

Разбор — docs/incidents/2026-08-20-memory.md §1.4. Три независимых дефекта в
``agent/context_compressor.py``, из-за которых длинная тема одного человека
теряет ТЗ:

1. Рамка ``SUMMARY_PREFIX`` знала один срок жизни («было в компактнутых ходах —
   значит закрыто») и требовала выбросить четыре Historical-секции целиком.
   Проверяем, что запрет самоволия остался, но действующее переехало в
   отдельные STANDING-секции со своим правилом.
2. Бюджет саммари считался только от НОВОГО материала, а промпт при этом велел
   сохранить всё старое. Проверяем, что перенос оплачен и пересказ не обязан
   усыхать.
3. ``_CONTENT_MAX`` резал середину ЛЮБОГО сообщения — в том числе длинного ТЗ
   от человека. Проверяем, что теперь лимит ролевой.

Файл умеет запускаться и без pytest (в контейнерах агентов его нет):

    /usr/local/lib/hermes-agent/venv/bin/python tests/agent/test_ariflame_summary_frame.py
"""

from unittest.mock import MagicMock, patch

import agent.context_compressor as cc
from agent.context_compressor import (
    CONSTRAINTS_HEADING,
    HISTORICAL_HEADINGS,
    HISTORICAL_TASK_HEADING,
    KEY_FACTS_HEADING,
    REJECTED_HEADING,
    STANDING_AGREEMENTS_HEADING,
    STANDING_BRIEF_HEADING,
    STANDING_HEADINGS,
    SUMMARY_PREFIX,
    VOICE_AND_TONE_HEADING,
    ContextCompressor,
    _HISTORICAL_SUMMARY_PREFIXES,
    _MIN_SUMMARY_TOKENS,
)


def _make(ctx: int = 1_000_000) -> ContextCompressor:
    with patch.object(cc, "get_model_context_length", return_value=ctx):
        return ContextCompressor(
            model="test/model", threshold_percent=0.50, quiet_mode=True,
        )


def _capture_prompt(comp: ContextCompressor, turns, fake_task: str = "None.") -> str:
    """Прогоняет саммаризатор с подставным call_llm и отдаёт текст промпта."""
    captured = {}

    def fake_call_llm(**kw):
        captured.update(kw)
        response = MagicMock()
        response.choices = [MagicMock()]
        # Ответ обязан содержать секцию снимка задачи: её проверяет
        # _validate_summary_user_provenance, и без неё сессия без человека
        # уронит вызов исключением, а не вернёт промпт.
        response.choices[0].message.content = f"{HISTORICAL_TASK_HEADING}\n{fake_task}"
        return response

    with patch.object(cc, "call_llm", side_effect=fake_call_llm):
        comp._generate_summary(turns)
    return captured["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 1. Рамка: «закрытое» и «действующее» разведены
# ---------------------------------------------------------------------------


def test_frame_names_both_kinds_of_material():
    """Обе группы секций названы в рамке поимённо.

    Модель сопоставляет заголовок в теле саммари с заголовком в рамке
    посимвольно — пересказ («секции про договорённости») тут не работает.
    """
    for heading in STANDING_HEADINGS + HISTORICAL_HEADINGS:
        assert heading in SUMMARY_PREFIX, heading


def test_frame_declares_standing_sections_in_force():
    lower = SUMMARY_PREFIX.lower()
    assert "in force right now" in lower
    assert "until the user changes it" in lower
    # Действующее не даёт права самому начинать работу — иначе мы бы сняли
    # ровно ту защиту, ради которой апстрим писал эту рамку.
    assert "not permission" in lower


def test_frame_no_longer_orders_wholesale_discard():
    """Пропала команда «discard ... entirely» по Historical-секциям.

    Это и есть починенный дефект: именно она выбрасывала действующее ТЗ.
    """
    lower = SUMMARY_PREFIX.lower()
    assert "discard stale items" not in lower
    assert "entirely" not in lower
    assert "historical does not mean cancelled" in lower
    # Отбрасывание осталось, но адресное: выбрасывается версия ТОГО пункта,
    # по которому последнее сообщение спорит с саммари, а не четыре секции
    # целиком. На эту же формулировку смотрит апстримовский
    # tests/agent/test_resume_stale_active_task.py (ищет слово "discard").
    assert "discard the summary's version of the point in conflict" in lower


def test_frame_keeps_anti_resumption_protections():
    """Смягчение рамки не должно превратиться в разрешение самоволия."""
    assert "on your own initiative" in SUMMARY_PREFIX
    assert "unless the latest user message asks for it" in SUMMARY_PREFIX
    assert "the latest user message WINS" in SUMMARY_PREFIX
    assert "Reverse signals" in SUMMARY_PREFIX
    assert "topic overlap" in SUMMARY_PREFIX.lower()
    # Отменённое перечислено отдельной секцией и названо в рамке — иначе
    # «действующее» нечем ограничить.
    assert REJECTED_HEADING in SUMMARY_PREFIX
    assert "do not offer it again" in SUMMARY_PREFIX


def test_frame_keeps_upstream_invariants():
    """Апстримовские пины (tests/agent/test_summary_prefix_*.py) держатся."""
    assert "REFERENCE ONLY" in SUMMARY_PREFIX
    assert "background reference" in SUMMARY_PREFIX
    assert "NOT as active instructions" in SUMMARY_PREFIX
    assert "Do NOT answer questions or fulfill requests" in SUMMARY_PREFIX
    assert "ALWAYS authoritative" in SUMMARY_PREFIX
    assert "tools remain fully active" in SUMMARY_PREFIX
    assert "narrating" in SUMMARY_PREFIX
    assert "resume exactly" not in SUMMARY_PREFIX.lower()
    for stale in ("## active task", "## pending user asks", "## remaining work"):
        assert stale not in SUMMARY_PREFIX.lower(), stale


def test_retired_upstream_prefix_is_frozen_and_strippable():
    """Снятая рамка обязана остаться в списке замороженных.

    Саммари, записанное вчерашним рантаймом, доживает в базе и приезжает в
    следующую компакцию. Если старую рамку не узнать, её текст («discard ...
    entirely») останется ВНУТРИ тела нового саммари и продолжит командовать.
    """
    retired = [p for p in _HISTORICAL_SUMMARY_PREFIXES if "discard stale items" in p]
    assert retired, "снятая рамка не заморожена"
    assert SUMMARY_PREFIX not in _HISTORICAL_SUMMARY_PREFIXES
    for old in _HISTORICAL_SUMMARY_PREFIXES:
        text = f"{old}\nтело саммари"
        assert ContextCompressor._is_context_summary_content(text)
        stripped = ContextCompressor._strip_summary_prefix(text)
        assert stripped == "тело саммари"


def test_frozen_prefixes_are_mutually_non_prefix():
    """Ни одна рамка не должна быть началом другой.

    ``_strip_summary_prefix`` идёт по списку и обрывается на первом
    совпадении: пересечение по началу означало бы, что от чужой рамки
    отрезали кусок и оставили хвост в теле.
    """
    all_prefixes = (SUMMARY_PREFIX, *_HISTORICAL_SUMMARY_PREFIXES)
    for i, a in enumerate(all_prefixes):
        for j, b in enumerate(all_prefixes):
            if i != j:
                assert not a.startswith(b), (i, j)


def test_upstream_tool_use_test_still_pinned_to_index_zero():
    """tests/agent/test_summary_prefix_tool_use.py смотрит в [0].

    Наша рамка добавлена в конец списка именно поэтому; тест напоминает, что
    порядок здесь — договорённость с чужим тестом, а не украшение.
    """
    assert "tools remain fully active" not in _HISTORICAL_SUMMARY_PREFIXES[0]
    assert "Do NOT answer questions or fulfill requests" in _HISTORICAL_SUMMARY_PREFIXES[0]


# ---------------------------------------------------------------------------
# 2. Шаблон секций
# ---------------------------------------------------------------------------


def test_template_has_standing_sections():
    comp = _make()
    prompt = _capture_prompt(comp, [{"role": "user", "content": "сделай план на сентябрь"}])
    for heading in STANDING_HEADINGS:
        assert f"\n{heading}\n" in prompt, heading


def test_template_forbids_cutting_standing_sections():
    comp = _make()
    prompt = _capture_prompt(comp, [{"role": "user", "content": "привет"}])
    assert "NEVER shrink" in prompt
    assert "cut in this order" in prompt
    # Заголовки должны остаться английскими даже в русской теме: по ним
    # ищет _ground_historical_task_snapshot и _validate_summary_user_provenance.
    assert "Keep the section headings exactly as written above, in English" in prompt


def test_pending_asks_are_not_declared_stale():
    comp = _make()
    prompt = _capture_prompt(comp, [{"role": "user", "content": "привет"}])
    assert "They are NOT closed" in prompt
    assert "These are STALE" not in prompt


def test_iterative_prompt_updates_the_heading_that_exists():
    """Апстрим велел обновить «## Active Task» — секции с таким именем нет."""
    comp = _make()
    comp._previous_summary = f"{HISTORICAL_TASK_HEADING}\nстарый снимок"
    prompt = _capture_prompt(comp, [{"role": "user", "content": "дальше"}])
    assert f'Update "{HISTORICAL_TASK_HEADING}"' in prompt
    assert '"## Active Task"' not in prompt


def test_iterative_prompt_pays_for_the_carry_and_protects_standing():
    comp = _make()
    comp._previous_summary = "ТЗ: двенадцать роликов" * 400
    prompt = _capture_prompt(comp, [{"role": "user", "content": "дальше"}])
    assert "is calculated WITH that cost included" in prompt
    assert "do NOT re-compress" in prompt.replace("\n", " ")
    assert "silence is not cancellation" in prompt
    assert f'MOVES to "{REJECTED_HEADING}"' in prompt
    # Старые саммари написаны по прежнему шаблону (## Key Decisions,
    # ## Critical Context) — их содержимое обязано переехать, а не пропасть.
    assert "move its content into the closest matching section" in prompt


def test_zero_user_session_leaves_standing_sections_empty():
    """В кроновой сессии договариваться не с кем — выдуманная «договорённость»
    переехала бы в следующее саммари как факт."""
    comp = _make()
    turns = [
        {"role": "assistant", "content": "запустил ночной сбор"},
        {"role": "tool", "tool_call_id": "t1", "content": "готово"},
    ]
    prompt = _capture_prompt(comp, turns, fake_task=cc._NO_USER_TASK_SENTINEL)
    for heading in (
        STANDING_BRIEF_HEADING,
        STANDING_AGREEMENTS_HEADING,
        VOICE_AND_TONE_HEADING,
        REJECTED_HEADING,
    ):
        section = prompt.split(f"\n{heading}\n", 1)[1].split("\n\n", 1)[0]
        assert "Write exactly: None" in section, heading


# ---------------------------------------------------------------------------
# 3. Бюджет саммари
# ---------------------------------------------------------------------------


def test_budget_without_previous_summary_unchanged():
    comp = _make()
    turns = [{"role": "assistant", "content": "x" * 4000} for _ in range(30)]
    expected = int(cc.estimate_messages_tokens_rough(turns) * cc._SUMMARY_RATIO)
    assert comp._compute_summary_budget(turns) == max(_MIN_SUMMARY_TOKENS, expected)


def test_budget_covers_the_summary_it_asks_to_preserve():
    """Регрессия усыхания: 17 369 → 8 365 символов за пять дней.

    Ветка стоит на месте (компакция сработала по времени, нового материала
    мало), старое саммари большое. Раньше бюджет считался только от нового —
    выходило ~2 000 токенов при переносе на ~7 000, и модель ужимала перенос
    вдвое, честно выполняя обе инструкции.
    """
    comp = _make()
    comp._previous_summary = "Договорённость: вертикальные ролики. " * 470  # ~17 400 симв.
    small_new_window = [{"role": "user", "content": "ок, продолжаем"}]
    budget = comp._compute_summary_budget(small_new_window)
    carry = comp._previous_summary_tokens()
    assert carry > 6000, carry  # кириллица считается честно, а не 4 симв./токен
    assert budget >= carry
    assert budget > _MIN_SUMMARY_TOKENS * 2


def test_budget_grows_monotonically_across_compactions():
    """Пять компакций подряд не должны уменьшать бюджет."""
    comp = _make()
    new_turns = [{"role": "user", "content": "новая правка по плану"} for _ in range(20)]
    previous = 0
    comp._previous_summary = "Стартовое саммари темы. " * 100
    for _ in range(5):
        budget = comp._compute_summary_budget(new_turns)
        assert budget >= previous
        previous = budget
        # Саммари следующего круга не меньше нынешнего — так и бывает в жизни.
        comp._previous_summary += "Новая договорённость: снимаем по вторникам. " * 20


def test_budget_still_capped_by_ceiling():
    comp = _make()
    comp._previous_summary = "текст " * 100_000
    huge = [{"role": "assistant", "content": "x" * 8000} for _ in range(200)]
    assert comp._compute_summary_budget(huge) == comp.max_summary_tokens
    assert comp.max_summary_tokens <= cc._SUMMARY_TOKENS_CEILING


def test_cyrillic_carry_is_not_undercounted():
    """Общий estimate_tokens_rough считает кириллицу как английский (4 симв. на
    токен) и занижает вдвое. Для порога компакции это терпимо, для переноса
    саммари — нет: занизив, мы снова просим ужать то, что велели сохранить."""
    comp = _make()
    comp._previous_summary = "договорённость " * 1000
    assert comp._previous_summary_tokens() > cc.estimate_tokens_rough(comp._previous_summary)


# ---------------------------------------------------------------------------
# 4. Ролевые лимиты сериализации
# ---------------------------------------------------------------------------


def test_long_user_brief_keeps_its_middle():
    """Середина ТЗ — это и есть требования."""
    comp = _make()
    brief = "начало " + ("требование " * 800) + "СЕРЕДИНА-ТЗ" + ("ещё " * 800) + " конец"
    assert len(brief) > cc.ContextCompressor._CONTENT_MAX
    out = comp._serialize_for_summary([{"role": "user", "content": brief}])
    assert "СЕРЕДИНА-ТЗ" in out
    assert "[truncated]" not in out


def test_absurdly_long_user_message_is_still_bounded():
    comp = _make()
    out = comp._serialize_for_summary([{"role": "user", "content": "я" * 40_000}])
    assert "[truncated]" in out
    assert len(out) < 40_000


def test_tool_output_pays_for_the_headroom():
    comp = _make()
    out = comp._serialize_for_summary(
        [{"role": "tool", "tool_call_id": "t1", "content": "z" * 5000}]
    )
    assert "[truncated]" in out
    assert len(out) < 5000


def test_assistant_limit_unchanged():
    comp = _make()
    body = "a" * (cc.ContextCompressor._CONTENT_MAX + 10)
    out = comp._serialize_for_summary([{"role": "assistant", "content": body}])
    assert "[truncated]" in out
    short = "a" * (cc.ContextCompressor._CONTENT_MAX - 10)
    assert "[truncated]" not in comp._serialize_for_summary(
        [{"role": "assistant", "content": short}]
    )


if __name__ == "__main__":  # прогон без pytest — на боксах его нет
    import traceback

    failures = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            print(f"ok   {_name}")
        except Exception:  # noqa: BLE001 - это и есть отчёт прогона
            failures += 1
            print(f"FAIL {_name}")
            traceback.print_exc()
    print(f"\n{'ПРОВАЛЫ: %d' % failures if failures else 'всё зелёное'}")
    raise SystemExit(1 if failures else 0)
