import asyncio
import json
import threading
from contextlib import suppress

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import MemorySaver

from promptloop.agent import create_eval_agent
from promptloop.cli import (
    CHANGE_TOO_LARGE_MESSAGE,
    _decide_file_write_interrupt,
    _oversized_review_rejection,
    _render_prompt_diff,
    _stream_response,
)
from promptloop.tools.prompts import ApprovalGate, make_prompt_tools, prompt_approval_key


def _oversized_edit():
    current = "".join(f"old line {index}\n" for index in range(150))
    proposed = "".join(f"new line {index}\n" for index in range(150))
    return current, proposed


def test_oversized_file_edit_rejection_tells_model_to_split(tmp_path):
    current, proposed = _oversized_edit()
    (tmp_path / "large.txt").write_text(current)
    interrupt_value = {
        "action_requests": [
            {
                "name": "edit_file",
                "args": {"file_path": "/large.txt", "old_string": current, "new_string": proposed},
            }
        ]
    }

    result = asyncio.run(_decide_file_write_interrupt(tmp_path, interrupt_value, True))

    decision = result["decisions"][0]
    assert decision["type"] == "reject"
    assert "too large to review" in decision["message"]
    assert "split" in decision["message"]
    assert "user declined" not in decision["message"]


def test_oversized_prompt_edit_rejection_reaches_model(tmp_path):
    current, proposed = _oversized_edit()
    edits = [{"old": current, "new": proposed}]
    gate = ApprovalGate(listener_timeout=0.01)
    tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, gate)}
    tools["register_prompt"].invoke({"source": current, "prompt_id": "p"})

    diff_error = _render_prompt_diff(tmp_path, "p", edits)
    assert diff_error == CHANGE_TOO_LARGE_MESSAGE
    rejection = _oversized_review_rejection(diff_error)
    assert rejection is not None

    key = prompt_approval_key("p", current, edits)
    gate.resolve(key, False, rejection)
    result = asyncio.run(tools["edit_prompt"].ainvoke({"prompt_id": "p", "edits": edits}))

    assert "too large to review" in result
    assert "split" in result
    assert (tmp_path / ".evals/prompts/p/current.txt").read_text() == current


def _blocking_second_response_model(responses, second_started, release_second):
    class BlockingSecondResponseChatModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _should_stream(self, *, async_api, run_manager=None, **kwargs):
            return True

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            if self.i == 1:
                second_started.set()
                release_second.wait(timeout=5)
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            message = result.generations[0].message
            content = message.content if isinstance(message.content, str) else ""
            tool_call_chunks = [
                {
                    "name": call["name"],
                    "args": json.dumps(call["args"]),
                    "id": call.get("id"),
                    "index": index,
                    "type": "tool_call_chunk",
                }
                for index, call in enumerate(message.tool_calls)
            ]
            if content or tool_call_chunks:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content, id=message.id, tool_call_chunks=tool_call_chunks)
                )
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    return BlockingSecondResponseChatModel(responses=responses)


def test_stream_response_waits_for_graph_after_midstream_approval(tmp_path):
    second_started = threading.Event()
    release_second = threading.Event()
    edits = [{"old": "before", "new": "after"}]
    setup_gate = ApprovalGate(listener_timeout=0.01)
    setup_tools = {tool.name: tool for tool in make_prompt_tools(tmp_path, setup_gate)}
    setup_tools["register_prompt"].invoke({"source": "before content", "prompt_id": "p"})

    async def scenario():
        gate = ApprovalGate(listener_timeout=1.0)
        tool_call = {
            "name": "edit_prompt",
            "args": {"prompt_id": "p", "edits": edits},
            "id": "call_edit_prompt",
            "type": "tool_call",
        }
        model = _blocking_second_response_model(
            [AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="done")],
            second_started,
            release_second,
        )
        checkpointer = MemorySaver()
        agent = create_eval_agent(tmp_path, checkpointer=checkpointer, gate=gate, chat_model=model)
        stream_task = asyncio.create_task(
            _stream_response(agent, "edit the prompt", "stream-lifecycle", tmp_path, gate, auto_approve=True)
        )

        try:
            started = await asyncio.to_thread(second_started.wait, 2)
            assert started
            assert not stream_task.done()
            release_second.set()
            await asyncio.wait_for(stream_task, 3)
        finally:
            release_second.set()
            if not stream_task.done():
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task

        state = await agent.aget_state({"configurable": {"thread_id": "stream-lifecycle"}})
        messages = state.values["messages"]
        assert any(isinstance(message, ToolMessage) for message in messages)
        assert any(isinstance(message, AIMessage) and message.content == "done" for message in messages)

    asyncio.run(scenario())
