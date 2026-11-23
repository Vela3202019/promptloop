import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from promptloop.agent import create_eval_agent
from promptloop.cli import (
    _decide_file_write_interrupt,
    _read_approval_key,
    _render_file_write_action,
    _stream_pending_file_writes,
    _stream_response,
)
from promptloop.tools import ApprovalGate


class FakeToolCallingChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


def _write_call(file_path="/out.txt", content="write gate content", call_id="call_write"):
    return {
        "name": "write_file",
        "args": {"file_path": file_path, "content": content},
        "id": call_id,
        "type": "tool_call",
    }


def _make_agent(project_dir, responses, checkpointer):
    model = FakeToolCallingChatModel(responses=responses)
    return create_eval_agent(project_dir, checkpointer=checkpointer, chat_model=model)


def _edit_call(file_path="/out.txt", old_string="before", new_string="after"):
    return {
        "name": "edit_file",
        "args": {"file_path": file_path, "old_string": old_string, "new_string": new_string},
        "id": "call_edit",
        "type": "tool_call",
    }


def _invoke_write_decision(tmp_path, decision):
    async def scenario():
        target = tmp_path / "out.txt"
        call = _write_call(content=f"content for {decision['type']}")
        agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")],
            MemorySaver(),
        )
        config = _config(f"write-{decision['type']}")

        result = await agent.ainvoke({"messages": [HumanMessage(content="write the file")]}, config=config)
        assert not target.exists()
        assert len(result["__interrupt__"]) == 1

        state = await agent.aget_state(config)
        assert len(state.interrupts) == 1
        request = state.interrupts[0].value["action_requests"][0]
        assert request["name"] == "write_file"
        assert request["args"]["file_path"] == "/out.txt"

        await agent.ainvoke(Command(resume={"decisions": [decision]}), config=config)
        state = await agent.aget_state(config)
        assert not state.interrupts
        return target.exists(), target.read_text() if target.exists() else None

    return asyncio.run(scenario())


def test_write_file_waits_for_approval_and_writes_after_approve(tmp_path):
    exists, content = _invoke_write_decision(tmp_path, {"type": "approve"})

    assert exists
    assert content == "content for approve"


def test_write_file_rejection_does_not_write(tmp_path):
    exists, content = _invoke_write_decision(
        tmp_path,
        {"type": "reject", "message": "The user declined the write."},
    )

    assert not exists
    assert content is None


def test_multiple_write_actions_in_one_interrupt(tmp_path):
    async def scenario():
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        calls = [
            _write_call("/first.txt", "first", "call_first"),
            _write_call("/second.txt", "second", "call_second"),
        ]
        agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=calls), AIMessage(content="done")],
            MemorySaver(),
        )
        config = _config("multiple-actions")

        await agent.ainvoke({"messages": [HumanMessage(content="write two files")]}, config=config)
        assert not first.exists()
        assert not second.exists()

        state = await agent.aget_state(config)
        assert len(state.interrupts) == 1
        assert len(state.interrupts[0].value["action_requests"]) == 2

        await agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}),
            config=config,
        )
        return first.read_text(), second.read_text()

    assert asyncio.run(scenario()) == ("first", "second")


def test_mixed_decisions_in_one_interrupt(tmp_path):
    async def scenario():
        rejected_target = tmp_path / "rejected.txt"
        approved_target = tmp_path / "approved.txt"
        calls = [
            _write_call("/rejected.txt", "rejected", "call_rejected"),
            _write_call("/approved.txt", "approved", "call_approved"),
        ]
        agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=calls), AIMessage(content="done")],
            MemorySaver(),
        )
        config = _config("mixed-actions")

        await agent.ainvoke({"messages": [HumanMessage(content="write two files")]}, config=config)
        await agent.ainvoke(
            Command(resume={
                "decisions": [
                    {"type": "reject", "message": "The user declined the first write."},
                    {"type": "approve"},
                ]
            }),
            config=config,
        )

        assert not rejected_target.exists()
        assert approved_target.read_text() == "approved"

    asyncio.run(scenario())


def _invoke_edit_decision(project_dir, decision):
    async def scenario():
        project_dir.mkdir()
        target = project_dir / "out.txt"
        target.write_text("before")
        agent = _make_agent(
            project_dir,
            [AIMessage(content="", tool_calls=[_edit_call()]), AIMessage(content="done")],
            MemorySaver(),
        )
        config = _config(f"edit-{decision['type']}")

        result = await agent.ainvoke({"messages": [HumanMessage(content="edit the file")]}, config=config)
        assert target.read_text() == "before"
        assert len(result["__interrupt__"]) == 1
        request = result["__interrupt__"][0].value["action_requests"][0]
        assert request["name"] == "edit_file"

        await agent.ainvoke(Command(resume={"decisions": [decision]}), config=config)
        return target.read_text()

    return asyncio.run(scenario())


def test_edit_file_waits_for_approval_and_applies_after_approve(tmp_path):
    assert _invoke_edit_decision(tmp_path / "approve", {"type": "approve"}) == "after"


def test_edit_file_rejection_does_not_modify_file(tmp_path):
    decision = {"type": "reject", "message": "The user declined the edit."}
    assert _invoke_edit_decision(tmp_path / "reject", decision) == "before"


def test_read_file_is_not_gated(tmp_path):
    async def scenario():
        (tmp_path / "input.txt").write_text("readable content")
        call = {
            "name": "read_file",
            "args": {"file_path": "/input.txt"},
            "id": "call_read",
            "type": "tool_call",
        }
        agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")],
            MemorySaver(),
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="read the file")]}, config=_config("read-only"))

        assert not result.get("__interrupt__")
        assert any(
            getattr(message, "name", None) == "read_file" and "readable content" in str(message.content)
            for message in result["messages"]
        )

    asyncio.run(scenario())


def test_stream_response_loops_through_sequential_interrupts_with_yes(tmp_path):
    async def scenario():
        calls = [
            _write_call("/first.txt", "first", "call_first"),
            _write_call("/second.txt", "second", "call_second"),
        ]
        responses = [
            AIMessage(content="", tool_calls=[calls[0]]),
            AIMessage(content="", tool_calls=[calls[1]]),
            AIMessage(content="done"),
        ]
        agent = _make_agent(tmp_path, responses, MemorySaver())

        await _stream_response(
            agent,
            "write both files",
            "stream-loop",
            tmp_path,
            ApprovalGate(),
            auto_approve=True,
        )

        assert (tmp_path / "first.txt").read_text() == "first"
        assert (tmp_path / "second.txt").read_text() == "second"

    asyncio.run(scenario())


def test_stream_response_non_tty_denies_without_yes(tmp_path):
    async def scenario():
        agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=[_write_call()]), AIMessage(content="done")],
            MemorySaver(),
        )

        await _stream_response(agent, "write", "stream-deny", tmp_path, ApprovalGate())

        assert not (tmp_path / "out.txt").exists()

    asyncio.run(scenario())


def test_pending_interrupt_streams_resumed_turn(tmp_path, capsys):
    async def scenario():
        target = tmp_path / "out.txt"
        checkpointer = MemorySaver()
        first_agent = _make_agent(
            tmp_path,
            [AIMessage(content="", tool_calls=[_write_call()]), AIMessage(content="unreachable")],
            checkpointer,
        )
        config = _config("cleanup")
        await first_agent.ainvoke({"messages": [HumanMessage(content="write")]}, config=config)
        assert len((await first_agent.aget_state(config)).interrupts) == 1

        restarted_agent = _make_agent(tmp_path, [AIMessage(content="done")], checkpointer)
        handled = await _stream_pending_file_writes(restarted_agent, "cleanup", tmp_path, ApprovalGate())

        assert handled is True
        assert not target.exists()
        assert not (await restarted_agent.aget_state(config)).interrupts

    asyncio.run(scenario())
    output = capsys.readouterr().out
    assert "approval required" in output
    assert "done" in output


def test_approval_key_non_tty_denies_and_yes_approves():
    assert asyncio.run(_read_approval_key(False)) is False
    assert asyncio.run(_read_approval_key(True)) is True


def test_rendering_error_denies_even_with_yes(tmp_path):
    interrupt_value = {
        "action_requests": [
            {
                "name": "write_file",
                "args": {"file_path": "/../escape.txt", "content": "must not write"},
            }
        ]
    }

    result = asyncio.run(_decide_file_write_interrupt(tmp_path, interrupt_value, True))

    assert result["decisions"][0]["type"] == "reject"


def test_malformed_interrupt_action_denies_without_raising(tmp_path):
    interrupt_value = {"action_requests": [{"name": "write_file", "args": "not-a-dict"}]}

    result = asyncio.run(_decide_file_write_interrupt(tmp_path, interrupt_value, True))

    assert result["decisions"] == [{
        "type": "reject",
        "message": "The pending file-write action was malformed; the file was not changed.",
    }]


def test_edit_file_preview_renders_diff(tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("alpha beta\n")

    error = _render_file_write_action(
        tmp_path,
        "edit_file",
        {"file_path": "/existing.txt", "old_string": "alpha", "new_string": "omega"},
    )

    assert error is None
