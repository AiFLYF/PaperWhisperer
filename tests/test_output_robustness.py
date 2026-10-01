"""What the output layer must refuse, and what it must never do.

A parser's happy-path tests are the easy half and prove very little: ``json.loads``
already handles a well-formed object, and a test that feeds one in is testing the
standard library. The half that matters is the half where a reply is not what the
prompt asked for, and this file is that half.

Four properties are asserted throughout, and the file is organised around them
rather than around the code's own structure, because a test suite organised like
the module cannot fail in a way the module did not anticipate:

1. **It never raises.** Every entry point is total, so a bad reply can never
   become a 500 no matter which of them a call site reached for.
2. **It always names what happened.** Every outcome is one of the closed
   :class:`~paperwhisperer.llm.output.OutputStatus` members, and a failure says
   what was expected and shows a bounded excerpt of what arrived.
3. **It never guesses.** No invented defaults, and in particular no empty
   container standing in for a payload that was not read -- that is the defect
   that makes an unreadable answer and a short answer look identical on screen.
4. **It is bounded.** Refusals are decided by measured ceilings, not by a
   judgement that something "feels fast".

**The live samples are committed, not fetched.** The fixtures under
:data:`LIVE_FENCED_QA` and friends were captured from the configured local
endpoint during the run that motivated this module, including its raw envelope
with ``finish_reason="tool_calls"``. They are pasted in here rather than
generated at test time so that the file passes with no endpoint, no network and
no credentials -- and so that a test asserting a specific reply cannot quietly
change meaning when the model behind it changes.

**The regression tests are the reason the other half is allowed to be clever.**
:class:`TestHappyPathIsUnchanged` pins the exact value every valid payload
produced *before this module existed*, computed by the same two lines the old
call sites used. Without it, a recovery that is too aggressive is indistinguishable
from a fix.
"""

from __future__ import annotations

import json
import time

import pytest

from paperwhisperer.core.errors import NonRetryableError
from paperwhisperer.core.text import extract_json_object
from paperwhisperer.llm import output as o
from paperwhisperer.llm.client import STREAM_FRAME_MAX_CHARS, LLMClient

# ---------------------------------------------------------------------------
# Live captures from the configured endpoint, committed as fixtures
# ---------------------------------------------------------------------------
#
# Taken during the run that produced this module. Reproduced verbatim, including
# the reasoning text, because the reasoning text is the evidence for *why* the
# refusal is not a retryable failure.

#: A tool call against a product that offers no tools, with an empty content.
#: Roughly half of all Q&A calls that invite a refusal arrive this way, per the
#: measurement recorded in ``tests/test_llm_integration.py``.
LIVE_TOOL_CALL_ENVELOPE = {
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": "",
                "reasoning_content": (
                    "The user wants a JSON object with keys \"answer\" and \"claims\". "
                    "But there's no context — no question was asked. I should look at "
                    "the working directory to see if there's relevant context, or just "
                    "ask for clarification.\n\nLet me check the environment first."
                ),
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"command": "pwd && ls -la"}',
                            "name": "bash",
                            "id": "call_function_cdq1evz6ac5l_1",
                            "type": "function",
                        }
                    },
                    {
                        "function": {
                            "arguments": '{"pattern": "**/*"}',
                            "name": "glob",
                            "id": "call_function_cdq1evz6ac5l_2",
                            "type": "function",
                        }
                    },
                ],
            },
        }
    ],
    "model": "space-bunny-free",
    "object": "chat.completion",
}

#: A perfectly good answer the model chose to fence. The default answer shape
#: for this endpoint, which is why a parser that cannot unwrap a fence loses
#: most of its answers rather than a rare minority.
LIVE_FENCED_QA = (
    "```json\n"
    "{\n"
    '  "answer": "本文使用的（并提出）模型是 FooNet-7B，即一个规模约 70 亿参数（7B）的模型。",\n'
    '  "claims": [\n'
    "    {\n"
    '      "text": "论文提出的是 FooNet-7B 模型，参数规模为 7B（约 70 亿）。",\n'
    '      "quote": "论文提出 FooNet-7B。",\n'
    '      "evidence_ids": ["ev_1"]\n'
    "    }\n"
    "  ]\n"
    "}"
    "```"
)

#: The same fact extraction without the fence.
LIVE_BARE_FACTS = (
    '{"datasets": [{"value": "ImageNet", "status": "extracted"}], '
    '"metrics": [{"value": "top-1"}]}'
)

#: Prose where a payload was expected, from a model that spent its whole budget
#: deciding to look for the document in the working directory. The second half
#: of that decision -- the empty ``content`` -- is :data:`LIVE_TOOL_CALL_ENVELOPE`.
LIVE_PROSE_REFUSAL = (
    "I'll analyze the text you provided to extract the requested information.\n\n"
    "Wait — I don't see any text included in your message. Your request specifies "
    '"根据这段文字抽取" (extract based on this passage of text), but the passage itself '
    "wasn't included — only the extraction target description and the desired output "
    "schema.\n\nLet me check whether there's relevant content in the working directory "
    "that you might be referring to:"
)

#: The payload the live prose refusal was trying to produce. Used to check that
#: a repair cannot rescue a refusal into a fabricated answer.
LIVE_FACTS_PAYLOAD = {
    "datasets": [{"value": "ImageNet", "status": "extracted"}],
    "metrics": [{"value": "top-1"}],
}

#: Payloads that were already valid JSON before this module existed. Every one
#: of these must parse to exactly the value the old call sites computed.
VALID_PAYLOADS = {
    "fact_extraction": (
        '{"research_question": {"value": "Q", "status": "extracted"},'
        ' "datasets": [{"value": "ImageNet", "status": "extracted"}]}'
    ),
    "qa_answer": json.dumps(
        {
            "answer": "an answer",
            "claims": [
                {"text": "a claim", "quote": "a quote", "evidence_ids": ["ev_1", "ev_2"]}
            ],
        },
        ensure_ascii=False,
    ),
    "search_rewrite": json.dumps(
        {"original_query": "q", "rewritten_query": "r", "topics": ["a", "b"], "why": "理由"}
    ),
    "appraisal": json.dumps(
        {"strengths": ["s"], "weaknesses": ["w"], "threats_to_validity": ["t"]},
        ensure_ascii=False,
    ),
    "empty_object": "{}",
    "empty_list": "[]",
    "chinese_value": '{"why": "他说“这是关键”，因为它就是关键。"}',
    "escaped_quotes": '{"a": "x\\" y", "b": "back\\\\slash"}',
    "unicode_escape": '{"a": "\\u4e2d\\u6587"}',
    "numbers": '{"a": 1, "b": -2.5, "c": 1e10, "d": 0}',
    "bools_null": '{"a": true, "b": false, "c": null}',
    "nested_deep": '{"a": {"b": {"c": {"d": [1, [2, [3, {"e": "f"}]]]}}}}',
    "whitespace_heavy": '  {\n\t"a"  :  1\n}  \n',
    "fullwidth_inside_string": '{"why": "准确率：95%，召回率：80%。"}',
}

#: Payloads the *old* ``extract_json_object`` helper could not read at all.
#: Not hypothetical: ``find("{")`` to ``rfind("}")`` on an array of objects
#: returns the two objects with the array brackets stripped, which is not JSON,
#: so every such reply became an empty result and a warning nobody read. They
#: are listed separately so the regression test below can assert the *right*
#: thing about them -- that the new layer reads them and the old one did not --
#: rather than quietly comparing against a broken baseline.
LEGACY_BROKEN_PAYLOADS = {
    "array_of_objects": '[{"value": "a"}, {"value": "b"}]',
    "two_objects_side_by_side": '{"a": 1} and {"b": 2}',
}

#: Structural punctuation that is ASCII syntax and full-width typography at the
#: same time. Kept as named constants so a test that means "a full-width colon
#: used as a key separator" cannot quietly be written as "a colon".
FW = "："
LBRACE, RBRACE = "｛", "｝"
LBRACKET, RBRACKET = "［", "］"
FW_QUOTE = "＂"
COMMA = "，"
LDQUOTE, RDQUOTE = "“", "”"


# ---------------------------------------------------------------------------
# Property-style: every variant returns a named outcome and raises nothing
# ---------------------------------------------------------------------------


def _variants() -> list[tuple[str, str]]:
    """Malformed and marginal replies, as ``(name, text)`` pairs.

    Written out rather than generated, because the interesting cases are
    specific and a generator only produces the ones its author imagined. The
    names are asserted against the taxonomy in
    :meth:`TestEveryVariantIsNamed`, so a variant that starts behaving
    differently has to be re-classified deliberately.
    """
    good = '{"answer": "yes", "claims": [{"text": "t", "evidence_ids": ["ev_1"]}]}'
    return [
        # -- the model said something usable but not exactly what was asked for
        ("fenced", f"```json\n{good}\n```"),
        ("fenced_no_language", f"```\n{good}\n```"),
        ("fenced_tildes", f"~~~json\n{good}\n~~~"),
        ("fenced_with_trailing_prose", f"```json\n{good}\n```\nLet me know!"),
        ("fence_unclosed", f"```json\n{good}"),
        ("preamble", f"Sure! Here is the JSON you asked for:\n{good}"),
        ("preamble_and_fence", f"Here you go:\n```json\n{good}\n```"),
        ("trailing_prose", f"{good}\n\nI hope this helps! Let me know."),
        ("preamble_and_trailing", f"Answer:\n{good}\nDone."),
        ("preamble_quoting_a_fragment", f'As you said, {{"x": 0}} is zero. The answer: {good}'),
        # -- typography a model reaches for when it is writing CJK
        ("fullwidth_colon", '{"answer"' + FW + '"yes"}'),
        ("fullwidth_braces", LBRACE + '"answer": "yes"' + RBRACE),
        ("fullwidth_brackets", '{"a": ' + LBRACKET + '1，2' + RBRACKET + '}'),
        ("fullwidth_quote_delim", FW_QUOTE + "answer" + FW_QUOTE + ":" + FW_QUOTE + "yes" + FW_QUOTE),
        ("fullwidth_everything", LBRACE + FW_QUOTE + "a" + FW_QUOTE + FW + LBRACKET + "1" + COMMA + "2" + RBRACKET + RBRACE),
        ("smart_quotes_as_delimiters", "{" + LDQUOTE + "a" + RDQUOTE + ": 1}"),
        ("smart_quotes_in_value", '{"a": "他说“这是关键”"}'),
        ("single_quotes", "{'answer': 'yes'}"),
        ("single_quotes_nested", "{'a': 'b', 'c': ['d', 'e']}"),
        # -- the reply was cut off, usually by the token budget
        ("truncated_object", '{"answer": "yes"'),
        ("truncated_array", '{"claims": [{"text": "t"'),
        ("truncated_deep", '{"a": {"b": [1, 2'),
        ("truncated_string", '{"answer": "unterminated'),
        ("truncated_at_a_comma", '{"a": 1, "b": 2,'),
        # -- nothing arrived
        ("empty_string", ""),
        ("whitespace_only", "   \n\t  \n "),
        ("null_bytes", "\x00\x00\x00"),
        ("fence_around_nothing", "```json\n```"),
        ("fence_around_whitespace", "```\n   \n```"),
        # -- something arrived, and it is not JSON
        ("prose_only", "I cannot help with that request."),
        ("refusal_prose", "很抱歉，我无法完成这个请求。"),
        ("html_body", "<!DOCTYPE html><html><body>login</body></html>"),
        ("doubled_braces", '{{"a": 1}}'),
        ("doubled_brackets", "[[1, 2]]"),
        ("unbalanced_close_first", '}{"a": 1}'),
        ("only_commas", ",,,,,"),
        ("only_colons", ":::::"),
        ("bare_brace", "{"),
        ("bare_bracket", "["),
        ("trailing_comma", '{"a": 1,}'),
        ("nan", '{"a": NaN}'),
        ("single_quoted_key_word", "{answer: yes}"),
        ("padded_control_chars", '{"a": 1}\x00\x07'),
        # -- structurally hostile
        ("deeply_nested", "[" * 200 + "]" * 200),
        ("deeply_nested_unclosed", "[" * 200),
        ("deeply_nested_mixed", '{"a":' * 60 + '1' + '}' * 60),
        ("long_prose", "word " * 50_000),
        ("one_huge_string", '{"a": "' + "x" * 100_000 + '"}'),
        ("many_small_objects", '{"a": 1}' * 20_000),
        ("alternating_quotes", '"' * 50_000),
        ("backslashes", "\\" * 50_000),
        ("nested_quotes", '{"a": "' + '\\"' * 20_000 + '"}'),
    ]


VARIANTS = _variants()


class TestEveryVariantIsNamed:
    """Requirement 1 and 2, asserted over the whole variant set at once."""

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_never_raises(self, name, text):
        # The assertion is the absence of an exception. Every one of these
        # inputs has killed a naive parser at some point, and a caller that
        # forgot an except clause turns any of them into a 500.
        result = o.parse_json(text, label=name)
        assert isinstance(result, o.OutputResult)

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_status_is_a_named_member_of_the_closed_taxonomy(self, name, text):
        result = o.parse_json(text, label=name)
        assert isinstance(result.status, o.OutputStatus)
        assert result.status in tuple(o.OutputStatus)

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_a_failure_explains_itself_in_chinese_and_shows_an_excerpt(self, name, text):
        result = o.parse_json(text, label=name)
        if result.ok:
            return
        assert result.user_message(), f"{name} failed with no user-facing message"
        if text.strip():
            # An excerpt of an input that had nothing in it would be an excerpt
            # of the empty string, so this is asserted only where there was
            # something to show.
            assert result.excerpt, f"{name} failed with no excerpt of what arrived"
            assert len(result.excerpt) <= o.EXCERPT_CHARS + 1

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_a_failure_carries_no_value(self, name, text):
        # The core "never guess" property. A failure that handed back an empty
        # container would let a call site render an unreadable reply as a section
        # that is present and empty, which is indistinguishable on screen from a
        # paper that says nothing.
        result = o.parse_json(text, label=name)
        if not result.ok:
            assert result.value is None

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_recovery_is_reported_never_implicit(self, name, text):
        # Anything that repaired bytes must say so, and must not claim to be a
        # clean parse. This is the property the whole module exists for.
        #
        # A *failure* is also allowed to carry a recovery tuple: that records
        # which repairs were attempted before the reading failed, which is the
        # only evidence a reader gets about why the bytes were unreadable. So
        # the two directions are asserted separately.
        result = o.parse_json(text, label=name)
        if result.ok:
            assert (bool(result.recoveries)) == (result.status is o.OutputStatus.RECOVERED), name
        if result.status is o.OutputStatus.OK:
            assert result.recoveries == (), name
        if result.recoveries:
            assert result.degradations()

    @pytest.mark.parametrize(("name", "text"), VARIANTS, ids=[item[0] for item in VARIANTS])
    def test_parsing_is_deterministic(self, name, text):
        # Determinism is what makes any of this testable: the same bytes must
        # always produce the same verdict, so a failure in CI and a failure on a
        # laptop are the same failure.
        first = o.parse_json(text, label=name)
        second = o.parse_json(text, label=name)
        assert first.to_dict() == second.to_dict()


class TestNonStringInputs:
    """The layer is typed ``Any`` because call sites pass whatever they have."""

    @pytest.mark.parametrize("value", [None, 0, 1.5, True, [], {}, (), object()])
    def test_never_raises_for_any_python_object(self, value):
        assert isinstance(o.parse_json(value).status, o.OutputStatus)

    def test_a_bytes_reply_is_read_as_text(self):
        result = o.parse_json(b'{"a": 1}')
        assert result.ok and result.value == {"a": 1}


# ---------------------------------------------------------------------------
# The taxonomy, one member at a time
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_the_set_is_closed_and_seven_strong(self):
        # A closed set is the mechanism. An open one means every `if status ==`
        # chain is a place where two authors can decide differently.
        assert {item.value for item in o.OutputStatus} == {
            "ok",
            "recovered",
            "not_json",
            "wrong_shape",
            "empty",
            "oversized",
            "refused",
        }

    def test_every_status_has_a_distinct_chinese_message_or_is_success(self):
        messages = {}
        for status in o.OutputStatus:
            result = o.OutputResult(status=status, detail="chars" if status is o.OutputStatus.OVERSIZED else "")
            message = result.user_message()
            if status is o.OutputStatus.OK:
                assert message == ""
                continue
            assert message
            assert message not in messages, f"{status} reuses {messages.get(message)!r}"
            messages[message] = status

    def test_empty_and_not_json_are_different_verdicts(self):
        # The distinction the brief is for: "said nothing useful" and "said
        # something we could not read" need different words on screen.
        assert o.parse_json("").status is o.OutputStatus.EMPTY
        assert o.parse_json("I cannot help.").status is o.OutputStatus.NOT_JSON
        assert o.parse_json("").status is not o.parse_json("I cannot help.").status

    def test_a_valid_reply_is_ok_and_reports_no_recovery(self):
        result = o.parse_json('{"a": 1}')
        assert result.status is o.OutputStatus.OK
        assert result.recoveries == ()
        assert result.ok and not result.degraded
        assert result.user_message() == ""
        assert result.degradations() == ()

    def test_a_repaired_reply_is_not_ok(self):
        result = o.parse_json('```json\n{"a": 1}\n```')
        assert result.status is o.OutputStatus.RECOVERED
        assert result.ok  # a value did come back
        assert result.degraded  # and it is not one we can vouch for
        assert result.is_recovered

    def test_wrong_shape_keeps_the_value_but_breaks_the_contract(self):
        result = o.parse_json_object("[1, 2, 3]")
        assert result.status is o.OutputStatus.WRONG_SHAPE
        assert result.value == [1, 2, 3]
        assert result.kind == "array"
        assert "数组" in result.user_message()

    def test_a_shape_mismatch_prefers_a_later_reading_that_fits(self):
        # The preamble is a valid array; the payload after it is the object the
        # prompt asked for. Reporting the mismatch from the first reading would
        # report a problem this reply does not have.
        result = o.parse_json_object('Some numbers: [1, 2] and the answer: {"a": 1}')
        assert result.status is o.OutputStatus.RECOVERED
        assert result.value == {"a": 1}

    def test_a_scalar_where_an_object_was_asked_for_is_wrong_shape(self):
        assert o.parse_json_object("42").status is o.OutputStatus.WRONG_SHAPE

    def test_to_dict_carries_the_value_only_on_success(self):
        # So an analysis run record cannot persist a half-read structure.
        assert "value" not in o.parse_json("nope").to_dict()
        assert o.parse_json('{"a": 1}').to_dict()["value"] == {"a": 1}

    def test_to_dict_is_json_serialisable(self):
        for _, text in VARIANTS:
            json.dumps(o.parse_json(text).to_dict(), ensure_ascii=False)

    def test_the_label_is_carried_into_the_user_message(self):
        result = o.parse_json("nope", label="事实层抽取")
        assert "事实层抽取" in result.user_message()


# ---------------------------------------------------------------------------
# What each recovery does, and what it is not allowed to touch
# ---------------------------------------------------------------------------


class TestRecoveries:
    @pytest.mark.parametrize(
        ("text", "expected", "recovery"),
        [
            ('```json\n{"a": 1}\n```', {"a": 1}, o.Recovery.FENCE),
            ("Sure, here it is:\n{\"a\": 1}", {"a": 1}, o.Recovery.PREAMBLE),
            ("{\"a\": 1}\nLet me know!", {"a": 1}, o.Recovery.TRAILING),
            ('{"a"' + FW + "1}", {"a": 1}, o.Recovery.FULLWIDTH),
            (LBRACE + '"a"' + FW + "1" + RBRACE, {"a": 1}, o.Recovery.FULLWIDTH),
            ("{" + LDQUOTE + "a" + RDQUOTE + FW + "1}", {"a": 1}, o.Recovery.SMART_QUOTES),
            ("{'a': 1}", {"a": 1}, o.Recovery.SINGLE_QUOTES),
            ('{"a": 1', {"a": 1}, o.Recovery.CLOSED),
        ],
    )
    def test_each_recovery_produces_the_right_value_and_names_itself(
        self, text, expected, recovery
    ):
        result = o.parse_json(text)
        assert result.status is o.OutputStatus.RECOVERED
        assert result.value == expected
        assert recovery in result.recoveries

    def test_a_recovery_never_claims_to_be_a_clean_parse(self):
        result = o.parse_json("```json\n{\"a\": 1}\n```")
        assert result.status is not o.OutputStatus.OK
        assert result.degradations()

    def test_full_width_punctuation_inside_a_string_is_content_not_syntax(self):
        # The rule that makes folding safe at all. A Chinese value legitimately
        # contains a full-width colon and a full-width comma; rewriting those
        # would corrupt the paper's own characters to repair a syntax error that
        # was never there.
        value = "准确率" + FW + "95%" + COMMA + "召回率" + FW + "80%。"
        result = o.parse_json(json.dumps({"why": value}, ensure_ascii=False))
        assert result.status is o.OutputStatus.OK
        assert result.value == {"why": value}

    def test_curly_quotes_inside_a_string_are_content_not_delimiters(self):
        value = "他说“这是关键”"
        result = o.parse_json(json.dumps({"why": value}, ensure_ascii=False))
        assert result.status is o.OutputStatus.OK
        assert result.value == {"why": value}
        assert o.Recovery.SMART_QUOTES not in result.recoveries

    def test_a_preamble_quoting_a_paper_title_does_not_swallow_the_payload(self):
        # The value-position rule. A reply that opens with `他说“这是关键”，因此：`
        # must not open a string on the first curly quote and lose everything
        # after it.
        text = "他说“这是关键”，因此：\n" + '{"answer": "yes"}'
        result = o.parse_json_object(text)
        assert result.value == {"answer": "yes"}

    def test_a_closing_brace_folded_from_full_width_is_accepted(self):
        assert o.parse_json('{"a": 1' + RBRACE).value == {"a": 1}

    def test_recover_false_disables_every_repair(self):
        # For input that is not model prose. A corrupt SSE frame must not be
        # turned into a plausible one.
        assert o.parse_json('```json\n{"a": 1}\n```', recover=False).status is (
            o.OutputStatus.NOT_JSON
        )
        assert o.parse_json('{"a"' + FW + "1}", recover=False).status is (
            o.OutputStatus.NOT_JSON
        )

    def test_the_recovery_tuple_is_deduplicated(self):
        # Two routes reach CLOSED -- the scan notes it, the span branch notes it
        # again when appending closers -- and a message that says the same
        # thing twice reads as two problems.
        result = o.parse_json('{"a": 1')
        assert list(result.recoveries).count(o.Recovery.CLOSED) == 1

    def test_a_failure_reports_the_repairs_it_attempted(self):
        # Without this, a failure with an empty recovery list reads as "the
        # model sent prose", when the truth is "the model sent JSON we could not
        # repair" -- and those send an operator to different places.
        result = o.parse_json(LBRACE + '"a": 1, ]')
        assert result.status is o.OutputStatus.NOT_JSON
        assert o.Recovery.FULLWIDTH in result.recoveries

    def test_plain_prose_reports_no_recovery(self):
        # The counterpart to the test above: prose really was prose.
        assert o.parse_json("I cannot help.").recoveries == ()


# ---------------------------------------------------------------------------
# The real failure modes of this product
# ---------------------------------------------------------------------------


class TestRealCaptures:
    """Replies actually returned by the configured endpoint."""

    def test_the_fenced_answer_parses_to_the_object_it_contained(self):
        result = o.parse_json_object(LIVE_FENCED_QA, label="Q&A")
        assert result.status is o.OutputStatus.RECOVERED
        assert o.Recovery.FENCE in result.recoveries
        assert result.value["answer"].startswith("本文使用的")
        assert result.value["claims"][0]["evidence_ids"] == ["ev_1"]
        assert result.value["claims"][0]["quote"] == "论文提出 FooNet-7B。"

    def test_the_unfenced_answer_needs_no_recovery(self):
        result = o.parse_json_object(LIVE_BARE_FACTS, label="事实层抽取")
        assert result.status is o.OutputStatus.OK
        assert result.value == LIVE_FACTS_PAYLOAD

    def test_the_tool_call_envelope_is_refused_not_empty(self):
        message = LIVE_TOOL_CALL_ENVELOPE["choices"][0]["message"]
        choice = LIVE_TOOL_CALL_ENVELOPE["choices"][0]
        verdict = o.classify_completion(
            message["content"],
            called_a_tool=bool(message["tool_calls"]),
            finish_reason=choice["finish_reason"],
            label="AI 回复",
        )
        assert verdict.status is o.OutputStatus.REFUSED
        assert verdict.detail == "tool_call"
        assert verdict.user_message() == o.TOOL_CALL_MESSAGE
        assert verdict.value is None

    def test_the_refusal_is_distinct_from_an_empty_reply(self):
        # The 500 this product used to produce. A refusal and an empty reply
        # have the same bytes and different meanings, and only one of them is
        # worth retrying.
        message = LIVE_TOOL_CALL_ENVELOPE["choices"][0]["message"]
        called = bool(message["tool_calls"])
        refused = o.classify_completion("", called_a_tool=called, finish_reason="tool_calls")
        empty = o.classify_completion("", called_a_tool=False, finish_reason="stop")
        assert refused.status is o.OutputStatus.REFUSED
        assert empty.status is o.OutputStatus.EMPTY
        assert refused.status is not empty.status
        assert refused.user_message() != empty.user_message()

    def test_the_layer_does_not_second_guess_the_client_about_a_tool_call(self):
        # The client's rule -- a tool call with no content is a refusal, not a
        # transport error -- is the layer's rule. They must not disagree, or
        # whichever one is wrong makes the same reply both retried and fatal.
        message = LIVE_TOOL_CALL_ENVELOPE["choices"][0]["message"]
        client_says_refused = bool(message["tool_calls"]) and not message["content"].strip()
        assert o.classify_completion(
            message["content"], called_a_tool=bool(message["tool_calls"])
        ).status is o.OutputStatus.REFUSED
        assert client_says_refused

    def test_a_reasoning_exhaustion_is_not_reported_as_an_empty_reply(self):
        # "length" with no content is the model spending the whole budget on
        # reasoning. Reporting it as "returned empty content" sends an operator
        # to debug the wrong thing entirely.
        verdict = o.classify_completion("", finish_reason="length")
        assert verdict.status is o.OutputStatus.EMPTY
        assert verdict.detail == "token_budget"
        assert verdict.detail != o.classify_completion("", finish_reason="stop").detail

    def test_the_prose_refusal_is_not_rescued_into_a_fabricated_answer(self):
        # The most important negative test here. This prose contains no payload,
        # and the payload the model was *aiming* for is not recoverable from it
        # -- the fact that we know what it wanted must not become what we claim
        # it said.
        result = o.parse_json_object(LIVE_PROSE_REFUSAL, label="Q&A")
        assert result.status is o.OutputStatus.NOT_JSON
        assert result.value is None
        assert result.value != LIVE_FACTS_PAYLOAD

    def test_the_client_still_refuses_the_live_tool_call_without_retrying(self, no_sleep):
        from .test_llm_client import FakeRawResponse, make_client

        class Reply:
            def __init__(self, choice):
                self.choices = [choice]

        class Choice:
            def __init__(self, message, finish_reason):
                self.message = message
                self.finish_reason = finish_reason

        class Message:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                self.reasoning_content = ""

        live = LIVE_TOOL_CALL_ENVELOPE["choices"][0]
        response = FakeRawResponse(None)
        response._completion = Reply(
            Choice(Message(live["message"]["content"], live["message"]["tool_calls"]),
                   live["finish_reason"])
        )
        instance = make_client(raw_behaviours=[response])
        # ``NonRetryableError``, not ``RuntimeError``: ``complete`` re-raises it
        # unwrapped precisely because a refusal is a final answer rather than a
        # blip, and the retry loop must not see it.
        with pytest.raises(NonRetryableError) as raised:
            instance.complete("s", "u", max_retries=3)
        assert "工具" in str(raised.value)
        assert str(raised.value) == o.TOOL_CALL_MESSAGE
        # Not retried: an identical prompt produces an identical tool call, and
        # three identical resends are how this became a 500.
        assert instance.client.raw_calls == 1


# ---------------------------------------------------------------------------
# The happy path is unchanged
# ---------------------------------------------------------------------------


class TestHappyPathIsUnchanged:
    """The regression that makes the rest of this file acceptable.

    Every value here is computed by the two lines the old call sites used --
    ``json.loads(extract_json_object(text))`` -- so a recovery that is too
    aggressive, a fold that hits a value it should not, or an ordering change
    that prefers a substring over the whole reply all show up here as a changed
    value rather than as a surprise in production.
    """

    @pytest.mark.parametrize("name", sorted(VALID_PAYLOADS))
    def test_a_valid_reply_parses_to_exactly_the_value_it_used_to(self, name):
        text = VALID_PAYLOADS[name]
        legacy = json.loads(extract_json_object(text))
        result = o.parse_json(text, label=name)
        assert result.value == legacy, name

    @pytest.mark.parametrize("name", sorted(VALID_PAYLOADS))
    def test_a_valid_reply_needs_no_recovery_and_says_so(self, name):
        result = o.parse_json(VALID_PAYLOADS[name], label=name)
        assert result.status is o.OutputStatus.OK, name
        assert result.recoveries == (), name
        assert not result.degraded

    @pytest.mark.parametrize("name", sorted(VALID_PAYLOADS))
    def test_the_kind_reported_matches_what_was_sent(self, name):
        result = o.parse_json(VALID_PAYLOADS[name], label=name)
        legacy = json.loads(extract_json_object(VALID_PAYLOADS[name]))
        assert result.kind == o._shape_of(legacy)

    def test_the_contract_helpers_agree_with_the_primitives(self):
        assert o.parse_json_object('{"a": 1}') == o.parse_json(
            '{"a": 1}', expected=o.EXPECT_OBJECT
        )
        assert o.parse_json_array("[1]") == o.parse_json("[1]", expected=o.EXPECT_ARRAY)

    def test_a_shape_mismatch_where_the_legacy_code_silently_emptied(self):
        # The behaviour being replaced, stated as a test. The old line was
        # `parsed if isinstance(parsed, Mapping) else {}`, which turns an array
        # and an unreadable reply into the same empty mapping -- and that is
        # what made a broken answer look like a paper that says nothing.
        legacy_mapping = {}
        parsed = json.loads(extract_json_object("[1, 2, 3]"))
        legacy_value = parsed if isinstance(parsed, dict) else legacy_mapping
        assert legacy_value == {}
        assert o.parse_json_object("[1, 2, 3]").status is o.OutputStatus.WRONG_SHAPE

    def test_an_empty_object_is_a_valid_answer_not_a_failure(self):
        # The fact layer's own contract: an empty object means "the model
        # returned nothing for any field", and that is a real answer that the
        # caller turns into eight recorded "not stated" facts. Folding it into
        # EMPTY would tell a user the model produced no output, which is a
        # different and wrong thing to say.
        result = o.parse_json_object("{}")
        assert result.status is o.OutputStatus.OK
        assert result.value == {}


class TestPayloadsTheOldHelperCouldNotRead:
    """A defect found by writing the regression test, and worth keeping fixed.

    ``extract_json_object`` locates the payload with ``find("{")`` and
    ``rfind("}")``. On a reply that is an *array* of objects it returns the
    objects with the brackets stripped, which is not JSON, so
    ``json.loads`` raises and the caller gets an empty section. On two objects
    side by side it returns both plus the prose between them.

    Both were live shapes: a fact field asked for as a list comes back as a list
    of objects constantly. They are asserted separately from the happy-path set
    because comparing against a baseline that raised would prove nothing --
    what is pinned here is that the new layer reads them correctly and reports
    honestly about the ones it can only partly read.
    """

    def test_an_array_of_objects_reads_correctly(self):
        result = o.parse_json_array(LEGACY_BROKEN_PAYLOADS["array_of_objects"])
        assert result.status is o.OutputStatus.OK
        assert result.value == [{"value": "a"}, {"value": "b"}]
        assert result.recoveries == ()

    def test_the_old_helper_really_did_fail_on_it(self):
        # If this ever stops failing the baseline has changed and the rest of
        # this class needs rechecking, so it is asserted rather than assumed.
        with pytest.raises(json.JSONDecodeError):
            json.loads(extract_json_object(LEGACY_BROKEN_PAYLOADS["array_of_objects"]))

    def test_the_first_object_of_a_prose_separated_pair_wins_and_says_why(self):
        # There is no honest way to pick between two payloads, and this module
        # does not pretend otherwise: it takes the first, and reports the
        # preamble/trailing repair so the reader knows a choice was made.
        result = o.parse_json_object(LEGACY_BROKEN_PAYLOADS["two_objects_side_by_side"])
        assert result.value == {"a": 1}
        assert result.status is o.OutputStatus.RECOVERED
        assert o.Recovery.TRAILING in result.recoveries


# ---------------------------------------------------------------------------
# Bounds, with measured ceilings
# ---------------------------------------------------------------------------


#: Ceilings for the pathological inputs, in seconds, for work that is
#: constant-time by construction. They are deliberately loose: the point is not
#: to measure the machine, it is to catch a change that turns a bounded refusal
#: into a linear or quadratic walk. A regression here is an order of magnitude,
#: not five percent.
#
#: The CPU-bound docling run this suite shares a host with argues for the loose
#: end of any plausible range as well.
PATHOLOGICAL_CEILING_SECONDS = 2.0


class TestBounds:
    def test_the_ceilings_are_ordered(self):
        # A depth ceiling above the size ceiling would mean the size check
        # always wins and the depth check is dead code.
        assert o.MAX_INPUT_CHARS < o.MAX_JSON_DEPTH * 10_000

    def test_ten_megabytes_of_braces_is_a_refusal_and_not_a_parse(self):
        # The pathological case from the brief, and the one a naive
        # ``json.loads`` pays full price for: it scans the whole thing before
        # discovering there is no value in it.
        payload = "{" * 10_000_000
        started = time.perf_counter()
        result = o.parse_json(payload)
        elapsed = time.perf_counter() - started
        assert result.status is o.OutputStatus.OVERSIZED
        assert result.detail == "chars"
        assert result.value is None
        assert elapsed < PATHOLOGICAL_CEILING_SECONDS, elapsed

    @pytest.mark.parametrize(
        "size", [o.MAX_INPUT_CHARS + 1, 1_000_000, 5_000_000, 10_000_000]
    )
    def test_the_oversize_refusal_is_constant_time_in_the_input(self, size):
        # The point of a bound: the work does not grow with the input. A
        # proportional cost here would be a denial-of-service vector that grows
        # with whatever a caller is willing to hold in memory. The sizes span
        # two orders of magnitude so a proportional cost cannot hide behind the
        # ceiling.
        payload = "{" * size
        started = time.perf_counter()
        assert o.parse_json(payload).status is o.OutputStatus.OVERSIZED
        assert time.perf_counter() - started < PATHOLOGICAL_CEILING_SECONDS

    def test_just_under_the_ceiling_still_parses(self):
        # A bound that refuses valid input is a bug in the other direction, and
        # one nobody notices until a long document will not analyse.
        value = "x" * 100_000
        text = json.dumps({"a": value}, ensure_ascii=False)
        assert len(text) < o.MAX_INPUT_CHARS
        assert o.parse_json(text).value == {"a": value}

    def test_just_over_the_ceiling_is_refused(self):
        result = o.parse_json("x" * (o.MAX_INPUT_CHARS + 1))
        assert result.status is o.OutputStatus.OVERSIZED
        assert result.detail == "chars"

    def test_the_ceiling_is_configurable_per_call(self):
        assert o.parse_json('{"a": 1}', max_chars=2).status is o.OutputStatus.OVERSIZED

    def test_deep_nesting_is_refused_rather_than_repaired(self):
        # ``json.loads`` recurses and raises ``RecursionError`` on this. An
        # exception from a parser is exactly what this module promises not to
        # produce, and the 2000-deep case is small enough to arrive by accident.
        result = o.parse_json("[" * 2000 + "]" * 2000)
        assert result.status is o.OutputStatus.OVERSIZED
        assert result.detail == f"depth={o.MAX_JSON_DEPTH}"
        assert result.value is None

    def test_nesting_within_the_ceiling_is_accepted(self):
        depth = o.MAX_JSON_DEPTH - 2
        result = o.parse_json("[" * depth + "]" * depth)
        assert result.status is o.OutputStatus.OK
        assert result.value is not None

    def test_the_depth_ceiling_is_configurable(self):
        assert o.parse_json("[" * 10, max_depth=5).status is o.OutputStatus.OVERSIZED

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("word " * 50_000, id="long_prose"),
            pytest.param('{"a": "' + "x" * 100_000 + '"}', id="one_huge_string"),
            pytest.param('{"a": 1}' * 20_000, id="many_objects"),
            pytest.param('"' * 50_000, id="alternating_quotes"),
            pytest.param("\\" * 50_000, id="backslashes"),
            pytest.param('{"a": "' + '\\"' * 20_000 + '"}', id="nested_escapes"),
            pytest.param("[" * 100_000, id="unclosed_brackets"),
        ],
    )
    def test_pathological_input_is_bounded(self, payload):
        # Each of these has a shape that makes a naive implementation work
        # harder rather than fail: no value to find, one enormous string,
        # thousands of siblings, and a quote or escape that never resolves.
        started = time.perf_counter()
        result = o.parse_json(payload)
        elapsed = time.perf_counter() - started
        assert isinstance(result.status, o.OutputStatus)
        assert elapsed < PATHOLOGICAL_CEILING_SECONDS, f"{elapsed:.3f}s"

    def test_repeated_smart_quotes_do_not_go_quadratic(self):
        # The specific shape that would catch a backward scan through the
        # output buffer: deciding "is this quote in value position" by searching
        # backwards instead of tracking the last significant character.
        payload = (LDQUOTE + "x" + COMMA) * 30_000
        started = time.perf_counter()
        o.parse_json(payload)
        assert time.perf_counter() - started < PATHOLOGICAL_CEILING_SECONDS

    def test_the_excerpt_is_bounded_whatever_arrives(self):
        result = o.parse_json("x" * (o.MAX_INPUT_CHARS + 1))
        assert len(result.excerpt) <= o.EXCERPT_CHARS + 1
        assert result.excerpt.endswith("…")

    def test_the_excerpt_collapses_newlines(self):
        # An excerpt is written to a log line; a multi-line one breaks the log.
        assert "\n" not in o.bounded_excerpt("a\nb\nc")
        assert o.bounded_excerpt("a\nb") == "a b"

    def test_the_excerpt_is_redacted(self):
        # It goes into a run record, and a model that echoes a prompt back can
        # echo a credential with it. ``sk-`` followed by 16+ characters is the
        # shape ``redact_secrets`` actually recognises.
        secret = "sk-abcdef0123456789abcdef"
        assert secret not in o.bounded_excerpt(json.dumps({"key": secret}))
        assert "[REDACTED]" in o.bounded_excerpt(json.dumps({"key": secret}))

    def test_the_excerpt_redacts_a_credential_the_process_itself_holds(
        self, monkeypatch
    ):
        # The literal pass, which is the one that cannot be evaded: it already
        # knows the value rather than having to recognise the shape.
        monkeypatch.setenv("OPENAI_API_KEY", "live-sentinel-value-xyz")
        excerpt = o.bounded_excerpt('the key is live-sentinel-value-xyz')
        assert "live-sentinel-value-xyz" not in excerpt
        assert "[REDACTED]" in excerpt


# ---------------------------------------------------------------------------
# The transport integration
# ---------------------------------------------------------------------------


class TestClientIntegration:
    def test_the_client_routes_a_stream_frame_through_the_layer(self):
        frame = LLMClient._parse_stream_frame('{"choices": [{"delta": {"content": "x"}}]}')
        assert frame == {"choices": [{"delta": {"content": "x"}}]}
        assert LLMClient._parse_stream_delta(frame) == "x"

    def test_an_unreadable_stream_frame_is_skipped(self):
        assert LLMClient._parse_stream_frame("data: {broken") is None
        assert LLMClient._parse_stream_frame("not json at all") is None

    def test_a_stream_frame_is_not_repaired(self):
        # A transport frame is not model prose. Repairing it would turn a
        # corrupt frame into a plausible one, and a stream that stops mid-word
        # for no visible reason is a bad afternoon to debug.
        assert LLMClient._parse_stream_frame('```json\n{"a": 1}\n```') is None
        assert LLMClient._parse_stream_frame('{"a"' + FW + "1}") is None

    def test_a_oversized_stream_frame_is_skipped(self):
        assert LLMClient._parse_stream_frame("{" * (STREAM_FRAME_MAX_CHARS + 1)) is None

    def test_a_frame_within_the_bound_still_parses(self):
        frame = LLMClient._parse_stream_frame(
            '{"choices": [{"delta": {"content": "' + "x" * 1000 + '"}}]}'
        )
        assert frame is not None

    def test_an_oversized_frame_is_skipped_where_a_stream_drops_it(self):
        from .test_llm_client import FakeStreamingResponse, make_client, sse_line

        oversize = "data: " + json.dumps(
            {"choices": [{"delta": {"content": "x" * (STREAM_FRAME_MAX_CHARS + 1)}}]}
        )
        response = FakeStreamingResponse(
            [oversize.encode("utf-8"), sse_line("still delivered")]
        )
        instance = make_client(stream_behaviours=[response])
        assert "".join(instance.stream("s", "u")) == "still delivered"

    def test_the_tool_call_message_is_owned_by_the_layer(self):
        # One owner for the wording. If the transport kept a private copy of
        # this string, the two would drift and one of them would be lying.
        assert "工具" in o.TOOL_CALL_MESSAGE
        assert "未提供内容回复" in o.TOOL_CALL_MESSAGE
