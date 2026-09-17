"""Small transport readers for the opt-in Rust frontend GPU smoke test.

Exhaust readers to validate successful framing. Closing one early is permitted
for cancellation tests, but deliberately does not certify a complete response.
"""

import json

PROMPT = "Write a short sentence about the ocean."


def make_body(api, stream, model, token_ids):
    assert api in ("text", "tokens", "completion", "chat"), api
    if api in ("text", "tokens"):
        return {
            "text" if api == "text" else "input_ids": (
                PROMPT if api == "text" else list(token_ids)
            ),
            "stream": stream,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        }
    body = {"model": model, "stream": stream, "temperature": 0, "max_tokens": 16}
    if api == "chat":
        body["messages"] = [{"role": "user", "content": PROMPT}]
    else:
        body["prompt"] = PROMPT
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _payload(raw):
    value = json.loads(raw)
    assert isinstance(value, dict), f"expected JSON object: {value!r}"
    assert "error" not in value, f"server error: {value!r}"
    return value


def http_frames(response, stream):
    """Read a requests-compatible response; full consumption validates EOF."""
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if not stream:
        assert content_type == "application/json", content_type
        yield _payload(response.content)
        return
    assert content_type == "text/event-stream", content_type
    data = []
    finished = False
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            if not data:
                continue
            raw = "\n".join(data)
            data.clear()
            assert not finished, "SSE data after [DONE]"
            if raw == "[DONE]":
                finished = True
            else:
                yield _payload(raw)
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        else:
            assert line.startswith(("event:", "id:", "retry:")), line
    assert not data, "unterminated SSE event"
    assert finished, "SSE response truncated before [DONE]"


def grpc_frames(call, api, stream):
    """Read synchronous runtime.v1 response iterators, including stream=false."""
    assert api in ("text", "tokens", "completion", "chat"), api
    native = api in ("text", "tokens")
    finished = False
    count = 0
    for chunk in call:
        assert not finished, "gRPC response after terminal marker"
        count += 1
        finished = chunk.finished
        if native:
            if not stream:
                assert finished and count == 1, (
                    "nonstream native must have one response"
                )
            value = {
                "meta_info": {
                    key: json.loads(raw) for key, raw in chunk.meta_info.items()
                }
            }
            value["text" if api == "text" else "output_ids"] = (
                chunk.text if api == "text" else list(chunk.output_ids)
            )
            assert bool(value["meta_info"].get("finish_reason")) == finished
            yield value
        elif stream:
            if finished:
                assert not chunk.json_chunk, "stream terminal must be empty"
            else:
                assert chunk.json_chunk, "empty nonterminal OpenAI chunk"
                yield _payload(chunk.json_chunk)
        else:
            assert finished and count == 1, (
                "nonstream OpenAI must have one terminal JSON chunk"
            )
            yield _payload(chunk.json_chunk)
    assert finished, "gRPC response truncated before terminal marker"


def _usage(value):
    assert isinstance(value, dict), f"missing usage: {value!r}"
    result = {
        key: value[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    assert all(type(count) is int and count > 0 for count in result.values()), result
    assert (
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    ), result
    return result


def summarize(api, frames, stream):
    """Return transport-stable output, finish_reason and usage for one request.

    Native output is text or token IDs; OpenAI output and finish_reason are
    lists ordered by choice index. Native streams must use cumulative output.
    """
    assert api in ("text", "tokens", "completion", "chat"), api
    frames = list(frames)
    assert frames, "no JSON response frames"
    if api in ("text", "tokens"):
        final = frames[-1]
        meta = final["meta_info"]
        finish = meta.get("finish_reason")
        assert isinstance(finish, dict) and finish.get("type") in ("stop", "length"), (
            finish
        )
        assert all(not frame["meta_info"].get("finish_reason") for frame in frames[:-1])
        counts = [frame["meta_info"]["completion_tokens"] for frame in frames]
        assert counts == sorted(counts), counts
        output = final["text" if api == "text" else "output_ids"]
        assert output, "empty native output"
        usage = _usage(
            {
                "prompt_tokens": meta["prompt_tokens"],
                "completion_tokens": meta["completion_tokens"],
                "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
            }
        )
        return {"output": output, "finish_reason": finish, "usage": usage}

    outputs, finishes = {}, {}
    usage = None
    for frame in frames:
        assert "error" not in frame, frame
        choices = frame["choices"]
        if frame.get("usage") is not None:
            assert usage is None, "multiple final usage records"
            usage = _usage(frame["usage"])
            if stream:
                assert choices == [], "expected final aggregate usage chunk"
        else:
            assert usage is None, "data after final usage"
        for choice in choices:
            index = choice["index"]
            assert type(index) is int and index >= 0, index
            assert index not in finishes, "choice data after finish"
            if api == "chat":
                text = choice["delta" if stream else "message"].get("content") or ""
            else:
                text = choice["text"]
            assert isinstance(text, str), text
            outputs[index] = outputs.get(index, "") + text
            reason = choice.get("finish_reason")
            if reason is not None:
                assert reason in ("stop", "length"), reason
                finishes[index] = reason
    indexes = sorted(outputs)
    assert indexes == list(range(len(indexes))) and indexes, indexes
    assert all(outputs[index] and index in finishes for index in indexes), (
        outputs,
        finishes,
    )
    assert usage is not None, "missing aggregate usage"
    if not stream:
        assert len(frames) == 1, "nonstream OpenAI returned multiple payloads"
    return {
        "output": [outputs[index] for index in indexes],
        "finish_reason": [finishes[index] for index in indexes],
        "usage": usage,
    }
