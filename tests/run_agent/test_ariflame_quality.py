"""ARIFLAME: модель качества для скиллов `model: quality` (agent/ariflame_quality.py).

Цикл гоняется настоящий (run_conversation), подменены только вызов модели и
выполнение тулов — так проверяется и место крючка, и запрос, который уйдёт в API.
"""

import json
import sys
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent import ariflame_quality as aq

LUNA = "gpt-5.6-luna"
TERRA = "gpt-5.6-terra"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (aq.ENV_MODEL, aq.ENV_EFFORT, aq.ENV_STICKY):
        monkeypatch.delenv(name, raising=False)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


def _quality_on(monkeypatch):
    monkeypatch.setenv(aq.ENV_MODEL, TERRA)


def _write_config_default(model=LUNA):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"model:\n  provider: openai-api\n  base_url: https://api.openai.com/v1\n  default: {model}\n",
        encoding="utf-8",
    )


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }


def _build_agent(monkeypatch, *, session_id="tg_42", model=LUNA, **kwargs):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [_tool("skill_view"), _tool("terminal")])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    agent = run_agent.AIAgent(
        model=model,
        provider="openai-api",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        quiet_mode=True,
        max_iterations=6,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
        **kwargs,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def _skill(tmp_path, name, *, quality=True, effort=None):
    d = tmp_path / "skills" / name
    d.mkdir(parents=True)
    head = [f"name: {name}", 'description: "test"']
    if quality:
        head.append("model: quality")
    if effort:
        head.append(f"effort: {effort}")
    (d / "SKILL.md").write_text("---\n" + "\n".join(head) + "\n---\n\n# Тело\n", encoding="utf-8")
    return d


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)


def _skill_call_response(args, *, model=LUNA):
    """Ход luna: зашифрованное рассуждение + реплика с id + вызов skill_view."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", id="rs_luna", encrypted_content="enc-luna",
                            summary=[], status="completed"),
            SimpleNamespace(type="message", id="msg_luna", role="assistant", status="completed",
                            content=[SimpleNamespace(type="output_text", text="Открываю скилл.")]),
            SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="skill_view",
                            arguments=json.dumps(args), status="completed"),
        ],
        usage=_usage(),
        status="completed",
        model=model,
    )


def _final_response(text="готово", *, model=TERRA):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", id="msg_final", role="assistant", status="completed",
                                content=[SimpleNamespace(type="output_text", text=text)])],
        usage=_usage(),
        status="completed",
        model=model,
    )


def _run(monkeypatch, agent, responses, *, skill_dir=None, skill_name="carousel-kit", history=None):
    requests = []

    def _api(api_kwargs):
        requests.append(api_kwargs)
        return responses.pop(0)

    def _exec(assistant_message, messages, effective_task_id, *_a):
        for call in assistant_message.tool_calls:
            payload = {"success": True, "name": skill_name, "content": "---\nname: x\n---\nтело"}
            if skill_dir is not None:
                payload["skill_dir"] = str(skill_dir)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(payload, ensure_ascii=False)})

    monkeypatch.setattr(agent, "_interruptible_api_call", _api)
    monkeypatch.setattr(agent, "_execute_tool_calls", _exec)
    result = agent.run_conversation("сделай карусель", conversation_history=history)
    return result, requests


def _types(items):
    return [it.get("type") or it.get("role") for it in items]


# ── выключено ───────────────────────────────────────────────────────────────

def test_off_when_env_unset(monkeypatch, tmp_path):
    agent = _build_agent(monkeypatch)
    sd = _skill(tmp_path, "carousel-kit", effort="high")
    result, requests = _run(monkeypatch, agent,
                            [_skill_call_response({"name": "carousel-kit"}), _final_response(model=LUNA)],
                            skill_dir=sd)

    assert result["completed"] is True
    assert [r["model"] for r in requests] == [LUNA, LUNA]
    replay = requests[1]["input"]
    reasoning = [it for it in replay if it.get("type") == "reasoning"]
    assert reasoning and reasoning[0]["encrypted_content"] == "enc-luna"
    assert any(it.get("id") == "msg_luna" for it in replay if it.get("type") == "message")
    # ни меток в истории, ни файла окна
    assert aq.STAMP not in json.dumps(result["messages"], ensure_ascii=False, default=str)
    assert not aq._store_path().exists()
    assert getattr(agent, "_ea_quality_base", None) is None


# ── переключение посреди прогона ────────────────────────────────────────────

def test_switch_on_quality_skill_view(monkeypatch, tmp_path):
    _quality_on(monkeypatch)
    agent = _build_agent(monkeypatch)
    sd = _skill(tmp_path, "carousel-kit", effort="high")
    result, requests = _run(monkeypatch, agent,
                            [_skill_call_response({"name": "carousel-kit"}), _final_response()],
                            skill_dir=sd)

    assert result["completed"] is True
    assert result["final_response"] == "готово"
    assert [r["model"] for r in requests] == [LUNA, TERRA]
    assert requests[1]["reasoning"]["effort"] == "high"  # effort из шапки скилла
    assert agent.model == TERRA
    assert agent._ea_quality_base["model"] == LUNA
    # промпт не пересобран: тот же, только строка Model: — новой модели
    assert f"Model: {LUNA}" in requests[0]["instructions"]
    assert f"Model: {TERRA}" in requests[1]["instructions"]
    assert requests[1]["instructions"].replace(TERRA, LUNA) == requests[0]["instructions"]

    replay = requests[1]["input"]
    # рассуждения luna терре не едут, id у реплики и вызова нет
    assert not [it for it in replay if it.get("type") == "reasoning"]
    msgs = [it for it in replay if it.get("type") == "message"]
    assert msgs and all("id" not in it for it in msgs)
    calls = [it for it in replay if it.get("type") == "function_call"]
    assert calls and all("id" not in it for it in calls)
    assert calls[0]["call_id"] == "call_1"
    assert aq.STAMP not in json.dumps(replay, ensure_ascii=False)

    # окно записано на сессию
    entry = aq.recall("tg_42")
    assert entry and entry["skill"] == "carousel-kit" and entry["effort"] == "high"


def test_effort_falls_back_to_env_then_medium(monkeypatch, tmp_path):
    _quality_on(monkeypatch)
    monkeypatch.setenv(aq.ENV_EFFORT, "low")
    agent = _build_agent(monkeypatch)
    sd = _skill(tmp_path, "deck")
    _, requests = _run(monkeypatch, agent,
                       [_skill_call_response({"name": "deck"}), _final_response()],
                       skill_dir=sd, skill_name="deck")
    assert requests[1]["model"] == TERRA
    assert requests[1]["reasoning"]["effort"] == "low"

    monkeypatch.delenv(aq.ENV_EFFORT)
    assert aq._default_effort() == "medium"


def test_no_switch_on_normal_skill(monkeypatch, tmp_path):
    _quality_on(monkeypatch)
    agent = _build_agent(monkeypatch)
    sd = _skill(tmp_path, "weather", quality=False)
    _, requests = _run(monkeypatch, agent,
                       [_skill_call_response({"name": "weather"}), _final_response(model=LUNA)],
                       skill_dir=sd, skill_name="weather")

    assert [r["model"] for r in requests] == [LUNA, LUNA]
    replay = requests[1]["input"]
    # своя модель — рассуждения едут как раньше, метки на провод не попадают
    reasoning = [it for it in replay if it.get("type") == "reasoning"]
    assert reasoning and reasoning[0]["encrypted_content"] == "enc-luna"
    assert any(it.get("id") == "msg_luna" for it in replay if it.get("type") == "message")
    assert aq.STAMP not in json.dumps(replay, ensure_ascii=False)
    assert not aq._store_path().exists()


def test_file_path_view_does_not_switch(monkeypatch, tmp_path):
    _quality_on(monkeypatch)
    agent = _build_agent(monkeypatch)
    sd = _skill(tmp_path, "carousel-kit")
    _, requests = _run(monkeypatch, agent,
                       [_skill_call_response({"name": "carousel-kit", "file_path": "references/a.md"}),
                        _final_response(model=LUNA)],
                       skill_dir=sd)
    assert [r["model"] for r in requests] == [LUNA, LUNA]


@pytest.mark.parametrize("session_id", ["guest_-100500", "group_-100777"])
def test_no_switch_in_guest_session(monkeypatch, tmp_path, session_id):
    _quality_on(monkeypatch)
    agent = _build_agent(monkeypatch, session_id=session_id)
    sd = _skill(tmp_path, "carousel-kit")
    _, requests = _run(monkeypatch, agent,
                       [_skill_call_response({"name": "carousel-kit"}), _final_response(model=LUNA)],
                       skill_dir=sd)
    assert [r["model"] for r in requests] == [LUNA, LUNA]
    assert not aq._store_path().exists()


@pytest.mark.parametrize("attr,value", [
    ("_delegate_depth", 1),                       # сабагент
    ("_memory_write_origin", "background_review"),  # фоновое ревью / куратор
    ("_chat_type", "group"),                      # группа родного гейтвея
    ("_gateway_session_key", "guest_-1"),
])
def test_not_eligible(monkeypatch, attr, value):
    _quality_on(monkeypatch)
    agent = _build_agent(monkeypatch)
    assert aq.eligible(agent) is True
    setattr(agent, attr, value)
    assert aq.eligible(agent) is False


def test_result_parser_tolerates_steer_tail_and_rejects_failures(tmp_path):
    sd = _skill(tmp_path, "site", effort="medium")
    ok = json.dumps({"success": True, "name": "site", "skill_dir": str(sd)})
    assert aq.quality_skill_from_result(ok + "\n\n[steer: и цвет поменяй]") == ("site", "medium")
    assert aq.quality_skill_from_result(json.dumps({"success": False, "error": "disabled"})) is None
    assert aq.quality_skill_from_result("<persisted-output>...") is None
    # без skill_dir — шапка из content
    inline = json.dumps({"success": True, "name": "x", "content": "---\nname: x\nmodel: quality\n---\n"})
    assert aq.quality_skill_from_result(inline) == ("x", None)


# ── история: запрос после смены модели ──────────────────────────────────────

def _history():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "старое"},
        {   # до включения фичи: без меток
            "role": "assistant", "content": "старый ответ",
            "codex_reasoning_items": [{"type": "reasoning", "id": "rs_old", "encrypted_content": "enc-old", "summary": []}],
            "codex_message_items": [{"type": "message", "role": "assistant", "id": "msg_old", "status": "completed",
                                     "content": [{"type": "output_text", "text": "старый ответ"}]}],
        },
        {"role": "user", "content": "карусель"},
        {
            "role": "assistant", "content": "",
            "codex_reasoning_items": [{"type": "reasoning", "id": "rs_l", "encrypted_content": "enc-luna",
                                       "summary": [], aq.STAMP: LUNA}],
            "codex_message_items": [{"type": "message", "role": "assistant", "id": "msg_l", "status": "completed",
                                     "content": [{"type": "output_text", "text": "Открываю"}], aq.STAMP: LUNA}],
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "skill_view", "arguments": "{\"name\": \"carousel-kit\"}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"success\": true}"},
        {
            "role": "assistant", "content": "слайд 1",
            "codex_reasoning_items": [{"type": "reasoning", "id": "rs_t", "encrypted_content": "enc-terra",
                                       "summary": [], aq.STAMP: TERRA}],
            "codex_message_items": [{"type": "message", "role": "assistant", "id": "msg_t", "status": "completed",
                                     "content": [{"type": "output_text", "text": "слайд 1"}], aq.STAMP: TERRA}],
        },
        {"role": "user", "content": "крупнее"},
    ]


def _payload(agent, history):
    kwargs = agent._build_api_kwargs(aq.portable_history(agent, history))
    return agent._get_transport().preflight_kwargs(kwargs, allow_stream=False, is_github_responses=False)


def test_payload_after_switch_drops_foreign_reasoning(monkeypatch):
    _quality_on(monkeypatch)
    _write_config_default(LUNA)
    agent = _build_agent(monkeypatch, model=TERRA)
    agent._ea_quality_base = {"model": LUNA, "reasoning_config": None, "config_context_length": None}
    history = _history()
    snapshot = json.dumps(history, ensure_ascii=False)

    payload = _payload(agent, history)
    items = payload["input"]
    wire = json.dumps(items, ensure_ascii=False)

    assert payload["model"] == TERRA
    assert [it["encrypted_content"] for it in items if it.get("type") == "reasoning"] == ["enc-terra"]
    ids = [it.get("id") for it in items if it.get("type") == "message"]
    assert ids == [None, None, "msg_t"]          # старая и luna — без id, своя — с id
    assert "enc-old" not in wire and "enc-luna" not in wire
    assert aq.STAMP not in wire
    assert json.dumps(history, ensure_ascii=False) == snapshot  # исходные сообщения не тронуты


def test_payload_on_box_model_keeps_own_and_legacy(monkeypatch):
    _quality_on(monkeypatch)
    _write_config_default(LUNA)
    agent = _build_agent(monkeypatch, model=LUNA)
    items = _payload(agent, _history())["input"]

    # luna: своё и старое (модели бокса) — едет, рассуждения терры — нет
    assert [it["encrypted_content"] for it in items if it.get("type") == "reasoning"] == ["enc-old", "enc-luna"]
    ids = [it.get("id") for it in items if it.get("type") == "message"]
    assert ids == ["msg_old", "msg_l", None]
    assert aq.STAMP not in json.dumps(items, ensure_ascii=False)


def test_portable_history_is_identity_when_off_and_unstamped(monkeypatch):
    agent = _build_agent(monkeypatch)
    history = [m for m in _history() if aq.STAMP not in json.dumps(m, ensure_ascii=False)]
    assert aq.portable_history(agent, history) is history


# ── окно между ходами ───────────────────────────────────────────────────────

def test_sticky_window_start_and_expiry_restore(monkeypatch):
    _quality_on(monkeypatch)
    monkeypatch.setenv(aq.ENV_STICKY, "120")
    now = [1_000_000.0]
    monkeypatch.setattr(aq, "time", SimpleNamespace(time=lambda: now[0]))
    aq.remember("tg_42", "carousel-kit", "high")

    # свежий агент (так API-сервер собирает каждый ход) — сразу на терре
    agent = _build_agent(monkeypatch)
    assert agent.model == LUNA
    base_reasoning = agent.reasoning_config
    _, requests = _run(monkeypatch, agent, [_final_response("слайд крупнее")])
    sticky_prompt = requests[0]["instructions"]
    assert f"Model: {TERRA}" in sticky_prompt
    assert requests[0]["model"] == TERRA
    assert requests[0]["reasoning"]["effort"] == "high"
    assert agent._ea_quality_base["model"] == LUNA

    # чужая сессия окна не видит
    other = _build_agent(monkeypatch, session_id="tg_43")
    _, requests = _run(monkeypatch, other, [_final_response(model=LUNA)])
    assert requests[0]["model"] == LUNA

    # тот же (закэшированный) агент через 121 минуту — назад на luna
    now[0] += 121 * 60
    _, requests = _run(monkeypatch, agent, [_final_response("привет", model=LUNA)])
    assert requests[0]["model"] == LUNA
    assert requests[0]["instructions"] == sticky_prompt.replace(TERRA, LUNA)
    assert agent.model == LUNA
    assert getattr(agent, "_ea_quality_base", None) is None
    assert agent.reasoning_config == base_reasoning
    assert agent._primary_runtime["model"] == LUNA


def test_restore_when_feature_turned_off(monkeypatch):
    _quality_on(monkeypatch)
    aq.remember("tg_42", "deck", "medium")
    agent = _build_agent(monkeypatch)
    aq.on_turn_start(agent)
    assert agent.model == TERRA

    monkeypatch.delenv(aq.ENV_MODEL)
    aq.on_turn_start(agent)
    assert agent.model == LUNA
    assert getattr(agent, "_ea_quality_base", None) is None


def test_manual_model_change_is_not_reverted(monkeypatch):
    _quality_on(monkeypatch)
    aq.remember("tg_42", "deck", "medium")
    agent = _build_agent(monkeypatch)
    aq.on_turn_start(agent)
    assert agent.model == TERRA
    agent.switch_model("gpt-5.6-sol", agent.provider, base_url=agent.base_url, api_mode=agent.api_mode)

    monkeypatch.setenv(aq.ENV_STICKY, "0")
    aq.on_turn_start(agent)
    assert agent.model == "gpt-5.6-sol"
    assert getattr(agent, "_ea_quality_base", None) is None


def test_store_prunes_expired_entries(monkeypatch):
    monkeypatch.setenv(aq.ENV_STICKY, "10")
    aq.remember("old", "deck", "medium", now=1000.0)
    aq.remember("new", "deck", "medium", now=1000.0 + 11 * 60)
    data = json.loads(aq._store_path().read_text(encoding="utf-8"))
    assert set(data) == {"new"}


def test_sticky_fresh_agent_reuses_stored_prompt(monkeypatch, tmp_path):
    """API-сервер: свежий агент на каждый ход, промпт — из базы. Окно не пересобирает
    промпт и не переписывает его в базе: там остаётся промпт модели бокса."""
    from hermes_state import SessionDB

    _quality_on(monkeypatch)
    db = SessionDB(db_path=tmp_path / "state.db")
    first = _build_agent(monkeypatch, session_db=db)
    first._persist_session = run_agent.AIAgent._persist_session.__get__(first)
    result, _ = _run(monkeypatch, first, [_final_response("привет", model=LUNA)])
    stored = db.get_session("tg_42")["system_prompt"]
    assert f"Model: {LUNA}" in stored

    aq.remember("tg_42", "deck", "medium")
    second = _build_agent(monkeypatch, session_db=db)
    history = [m for m in result["messages"] if m.get("role") != "system"]
    _, requests = _run(monkeypatch, second, [_final_response("слайд")], history=history)

    assert requests[0]["model"] == TERRA
    assert requests[0]["instructions"] == stored.replace(f"Model: {LUNA}", f"Model: {TERRA}")
    assert db.get_session("tg_42")["system_prompt"] == stored
