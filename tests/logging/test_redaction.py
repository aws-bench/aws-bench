"""Tests for ledger env capture and secret redaction helpers."""

from __future__ import annotations

import json

import pytest

from aws_bench.logging.redaction import (
    REDACTED,
    record_env,
    redact_argv,
    redact_config,
    redact_json,
)

# Numeric/boolean values under secret-named keys: all must be masked.
_SECRET_KEY_SCALARS = [
    "access_tokens=1234567890",
    "refresh_tokens=99999999",
    "GH_TOKENS=4096",
    "session_tokens=true",
    "accessTokenSecret=1234567890",
    "API_KEY=12345",
    "GH_TOKEN=7",
    "TOKEN=true",
]
# Benign count keys that contain a secret fragment: masked too (documented trade-off).
_OVER_MASKED = ["max_tokens=4096", "max_completion_tokens=1024", "count_tokens=false"]


class TestRedactArgv:
    def test_redacts_value_after_secret_named_flag(self):
        argv = ["aws-bench", "run", "--registry-token", "supersecretvalue123"]
        assert redact_argv(argv) == ["aws-bench", "run", "--registry-token", REDACTED]

    def test_redacts_equals_form_secret_flag(self):
        argv = ["aws-bench", "--api-token=supersecretvalue123"]
        assert redact_argv(argv) == ["aws-bench", "--api-token=" + REDACTED]

    def test_keeps_ordinary_flags_and_values(self):
        argv = ["aws-bench", "run", "-d", "awsbench@1.0.0"]
        assert redact_argv(argv) == argv

    def test_passes_plain_url_through(self):
        argv = ["aws-bench", "run", "--registry-url", "https://github.com/org/repo"]
        assert redact_argv(argv) == argv

    def test_does_not_redact_long_non_secret_value(self):
        argv = ["aws-bench", "run", "9f3c1ab2d4e5f60718293a4b5c6d7e8f90123456"]
        assert redact_argv(argv) == argv

    @pytest.mark.parametrize("flag", ["--ve", "--verifier-env", "--ae", "--agent-env"])
    def test_redacts_secret_env_assignment_after_env_flag(self, flag):
        argv = ["aws-bench", "run", flag, "AWS_BEARER_TOKEN_BEDROCK=CANARY-ENV"]
        expected = ["aws-bench", "run", flag, f"AWS_BEARER_TOKEN_BEDROCK={REDACTED}"]
        assert redact_argv(argv) == expected

    @pytest.mark.parametrize("flag", ["--ve", "--verifier-env", "--ae", "--agent-env"])
    def test_redacts_inline_secret_env_assignment(self, flag):
        argv = ["aws-bench", "run", f"{flag}=AWS_BEARER_TOKEN_BEDROCK=CANARY-INLINE"]
        expected = ["aws-bench", "run", f"{flag}=AWS_BEARER_TOKEN_BEDROCK={REDACTED}"]
        assert redact_argv(argv) == expected

    def test_inline_env_flag_does_not_redact_following_token(self):
        # The inline operand contains TOKEN; the next token must not be swallowed.
        argv = ["aws-bench", "run", "--ve=MY_TOKEN=CANARY-INLINE", "--model", "m"]
        expected = ["aws-bench", "run", f"--ve=MY_TOKEN={REDACTED}", "--model", "m"]
        assert redact_argv(argv) == expected

    @pytest.mark.parametrize(
        "key",
        [
            "MY_SECRET",
            "GH_TOKEN",
            "DB_PASSWORD",
            "X_API_KEY",
            "AWS_CREDENTIALS",
            "OAUTH_CLIENT",
            "BEARER",
        ],
    )
    def test_redacts_every_secret_key_fragment(self, key):
        argv = ["aws-bench", "run", "--ae", f"{key}=CANARY-FRAGMENT"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ae", f"{key}={REDACTED}"]

    def test_inline_non_secret_flag_with_secret_looking_value_is_kept(self):
        # Only the flag name decides; a value containing KEY must not trigger masking.
        argv = ["aws-bench", "run", "--agent-import-path=pkg.keystore:Agent", "-m", "model-x"]
        assert redact_argv(argv) == argv

    def test_secret_key_match_is_case_insensitive(self):
        argv = ["aws-bench", "run", "--ve", "my_token=CANARY-LOWER"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve", f"my_token={REDACTED}"]

    def test_keeps_non_secret_env_assignment(self):
        argv = ["aws-bench", "run", "--ve", "FOO=bar", "--ae", "AWS_REGION=us-east-1"]
        assert redact_argv(argv) == argv

    def test_keeps_defaultless_env_template_value(self):
        # A defaultless ``${VAR}`` carries no value; Harbor resolves it from the host env.
        argv = ["aws-bench", "run", "--ve", "AWS_BEARER_TOKEN_BEDROCK=${AWS_BEARER_TOKEN_BEDROCK}"]
        assert redact_argv(argv) == argv

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("${OTHER:-x}", "${OTHER:-***}"),
            ("${AWS_BEARER_TOKEN_BEDROCK:-}", "${AWS_BEARER_TOKEN_BEDROCK:-***}"),
            ("${MISSING:-CANARY-DEFAULT}", "${MISSING:-***}"),
            ("${MISSING:-a=b}", "${MISSING:-***}"),
        ],
    )
    def test_masks_env_template_default(self, value, expected):
        # Harbor falls back to the typed default when the host var is unset, so the
        # default is a literal value: mask it, keep the key and the variable name.
        argv = ["aws-bench", "run", "--ve", f"TOKEN={value}"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve", f"TOKEN={expected}"]

    def test_masks_inline_env_template_default(self):
        argv = ["aws-bench", "run", "--ve=X_TOKEN=${MISSING:-CANARY-INLINE-DEFAULT}"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve=X_TOKEN=${MISSING:-***}"]

    @pytest.mark.parametrize("value", ["${}", "${A:B}", "${A}${B}", "$A", "${A:-x}tail"])
    def test_masks_non_template_lookalikes(self, value):
        argv = ["aws-bench", "run", "--ve", f"MY_TOKEN={value}"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve", f"MY_TOKEN={REDACTED}"]

    def test_keeps_inline_env_template_value(self):
        argv = ["aws-bench", "run", "--ve=AWS_BEARER_TOKEN_BEDROCK=${AWS_BEARER_TOKEN_BEDROCK}"]
        assert redact_argv(argv) == argv

    def test_redacts_template_lookalike_with_trailing_text(self):
        # Only a whole-value template is kept; anything else is a literal value.
        argv = ["aws-bench", "run", "--ve", "MY_TOKEN=${X}CANARY-TAIL"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve", f"MY_TOKEN={REDACTED}"]

    def test_redacts_standalone_secret_assignment(self):
        argv = ["aws-bench", "run", "--ak", "X_API_KEY=CANARY-STANDALONE", "X_API_KEY=CANARY-2"]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--ak",
            f"X_API_KEY={REDACTED}",
            f"X_API_KEY={REDACTED}",
        ]

    def test_keeps_standalone_non_secret_assignment(self):
        argv = ["aws-bench", "run", "A=b", "--ak", "temperature=0.2"]
        assert redact_argv(argv) == argv

    def test_value_with_multiple_equals_is_fully_masked(self):
        argv = ["aws-bench", "run", "--ve", "TOKEN=a=b", "TOKEN=c=d"]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--ve",
            f"TOKEN={REDACTED}",
            f"TOKEN={REDACTED}",
        ]

    def test_secret_assignment_with_empty_value(self):
        argv = ["aws-bench", "run", "--ve", "MY_TOKEN=", "MY_TOKEN="]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--ve",
            f"MY_TOKEN={REDACTED}",
            f"MY_TOKEN={REDACTED}",
        ]

    @pytest.mark.parametrize("operand", ["", "NOEQUALS"])
    def test_odd_env_operands_do_not_raise(self, operand):
        argv = ["aws-bench", "run", "--ve", operand]
        assert redact_argv(argv) == argv

    @pytest.mark.parametrize("operand", ["=CANARY-NOKEY", "=", "==CANARY"])
    def test_empty_key_assignment_is_fully_masked(self, operand):
        # No valid invocation has an empty key, so the whole token is masked.
        assert redact_argv(["aws-bench", "run", "--ve", operand]) == [
            "aws-bench",
            "run",
            "--ve",
            REDACTED,
        ]
        assert redact_argv(["aws-bench", "run", f"--ve={operand}"]) == [
            "aws-bench",
            "run",
            f"--ve={REDACTED}",
        ]

    def test_blank_key_assignment_is_decided_by_key(self):
        # Only a strictly empty key is masked whole; a blank key is a benign name.
        assert redact_argv(["aws-bench", "run", "--ve", "   =X"])[-1] == "   =X"

    def test_url_with_secret_first_query_key_is_masked_fail_safe(self):
        # URLs get no carve-out in argv: everything after the first ``=`` goes.
        url = "https://host/p?api_key=CANARY-URL"
        masked = f"https://host/p?api_key={REDACTED}"
        argv = ["aws-bench", "run", "--registry-url", url, f"--mcp-url={url}", url]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--registry-url",
            masked,
            f"--mcp-url={masked}",
            masked,
        ]

    def test_inline_env_flag_with_empty_operand(self):
        argv = ["aws-bench", "run", "--ve=", "--ve=NOEQUALS"]
        assert redact_argv(argv) == argv

    def test_trailing_env_flag_without_operand(self):
        argv = ["aws-bench", "run", "--ve"]
        assert redact_argv(argv) == argv

    def test_env_flag_followed_by_flag_does_not_swallow_it(self):
        argv = ["aws-bench", "run", "--ve", "--api-key", "CANARY-NEXT"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ve", "--api-key", REDACTED]

    def test_empty_argv(self):
        assert redact_argv([]) == []

    def test_empty_string_tokens(self):
        assert redact_argv(["", ""]) == ["", ""]

    def test_secret_flag_and_env_assignment_together(self):
        argv = [
            "aws-bench",
            "run",
            "--registry-token",
            "CANARY-FLAG",
            "--ve",
            "AWS_BEARER_TOKEN_BEDROCK=CANARY-VE",
            "--api-token=CANARY-EQ",
        ]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--registry-token",
            REDACTED,
            "--ve",
            f"AWS_BEARER_TOKEN_BEDROCK={REDACTED}",
            f"--api-token={REDACTED}",
        ]

    @pytest.mark.parametrize(
        "flag",
        [
            "--ak",
            "--agent-kwarg",
            "--verifier-kwarg",
            "--ve",
            "--verifier-env",
            "--ae",
            "--agent-env",
        ],
    )
    @pytest.mark.parametrize(
        ("assignment", "expected"),
        [
            ("api_key=CANARY-KW", f"api_key={REDACTED}"),
            ("AUTH_HEADER=CANARY-KW", f"AUTH_HEADER={REDACTED}"),
            ("api_token=a=CANARY-KW", f"api_token={REDACTED}"),
            ("temperature=0.2", "temperature=0.2"),
        ],
    )
    def test_key_value_carrier_flags_spaced_and_inline(self, flag, assignment, expected):
        spaced = ["aws-bench", "run", flag, assignment, "--model", "m"]
        inline = ["aws-bench", "run", f"{flag}={assignment}", "--model", "m"]
        assert redact_argv(spaced) == ["aws-bench", "run", flag, expected, "--model", "m"]
        assert redact_argv(inline) == ["aws-bench", "run", f"{flag}={expected}", "--model", "m"]

    @pytest.mark.parametrize("form", ["spaced", "inline"])
    def test_metric_kwarg_first_assignment_is_masked(self, form):
        # ``--metric type:k=v`` - the text before the first ``=`` decides.
        value = "mean:api_key=CANARY-METRIC"
        argv = [
            "aws-bench",
            "run",
            *(["--metric", value] if form == "spaced" else [f"--metric={value}"]),
        ]
        out = redact_argv(argv)
        assert "CANARY-METRIC" not in " ".join(out)
        assert out[-1].endswith(f"mean:api_key={REDACTED}")

    def test_secret_named_inline_flag_masks_whole_value(self):
        argv = ["aws-bench", "--registry-token=FOO=CANARY-SECRET-FLAG"]
        assert redact_argv(argv) == ["aws-bench", f"--registry-token={REDACTED}"]

    def test_inline_benign_flag_with_benign_assignment_is_kept(self):
        argv = ["aws-bench", "run", "--ak=agent_engine=bedrock", "--ve=LOG_LEVEL=debug"]
        assert redact_argv(argv) == argv

    @pytest.mark.parametrize("assignment", _SECRET_KEY_SCALARS + _OVER_MASKED)
    def test_secret_keys_mask_numeric_and_bool_values(self, assignment):
        # No value-based exemption: every value under a secret key is masked,
        # spaced and inline, including benign count keys (accepted over-masking).
        key = assignment.partition("=")[0]
        argv = ["aws-bench", "run", "--ak", assignment, f"--ak={assignment}"]
        assert redact_argv(argv) == [
            "aws-bench",
            "run",
            "--ak",
            f"{key}={REDACTED}",
            f"--ak={key}={REDACTED}",
        ]

    @pytest.mark.parametrize(
        "assignment",
        [
            "max_tokens=sk-CANARY",
            "AWS_BEARER_TOKEN_BEDROCK=abc",
            "monkey=capuchin",  # documented trade-off of substring matching
            "KEYBOARD_LAYOUT=us",  # documented trade-off of substring matching
        ],
    )
    def test_secret_keys_mask_string_values(self, assignment):
        key = assignment.partition("=")[0]
        argv = ["aws-bench", "run", "--ak", assignment]
        assert redact_argv(argv) == ["aws-bench", "run", "--ak", f"{key}={REDACTED}"]

    def test_url_looking_assignment_value_is_masked(self):
        # The ``://`` carve-out looks only at the key part; this key is ``api_key``.
        argv = ["aws-bench", "run", "--ak", "api_key=https://CANARY-URL@h/x"]
        assert redact_argv(argv) == ["aws-bench", "run", "--ak", f"api_key={REDACTED}"]

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("API_KEY = sk-CANARY", f"API_KEY ={REDACTED}"),
            (" api_key=sk-CANARY", f" api_key={REDACTED}"),
            ("\tMY_TOKEN =CANARY", f"\tMY_TOKEN ={REDACTED}"),
        ],
    )
    def test_padded_assignment_is_masked_like_config(self, token, expected):
        # Same rule and output as the config string arm: key text kept verbatim.
        assert redact_argv(["aws-bench", "run", "--ve", token])[-1] == expected
        assert redact_argv([token]) == [expected]
        assert redact_json(token) == expected

    def test_bare_tag_token_is_masked_fail_safe(self):
        # ``Key`` is a secret-named key, so everything after the first ``=`` goes.
        assert redact_argv(["Key=Name,Value=web-1"]) == [f"Key={REDACTED}"]

    def test_benign_assignments_stay_visible(self):
        argv = ["aws-bench", "run", "--ve", "LOG_LEVEL=debug", "--ak", "temperature=0.2"]
        assert redact_argv(argv) == argv

    def test_input_list_is_not_mutated(self):
        argv = ["aws-bench", "run", "--ve", "MY_TOKEN=CANARY-MUT"]
        snapshot = list(argv)
        redact_argv(argv)
        assert argv == snapshot


class TestRedactConfig:
    def test_redacts_secret_named_field(self):
        out = redact_config({"registry_token": "abc123", "n_attempts": 4})
        assert out == {"registry_token": REDACTED, "n_attempts": 4}

    def test_recurses_into_nested_dicts_and_lists(self):
        out = redact_config({"agents": [{"api_key": "x", "name": "kiro"}]})
        assert out == {"agents": [{"api_key": REDACTED, "name": "kiro"}]}

    def test_leaves_plain_url_field_intact(self):
        out = redact_config({"registry_url": "https://github.com/org/repo"})
        assert out == {"registry_url": "https://github.com/org/repo"}

    def test_none_passes_through(self):
        assert redact_config(None) is None

    def test_redacts_auth_and_bearer_named_fields(self):
        out = redact_config(
            {"oauth_client": "c", "Authorization": "a", "bearer": "b", "auth_header": "h"}
        )
        assert out == {
            "oauth_client": REDACTED,
            "Authorization": REDACTED,
            "bearer": REDACTED,
            "auth_header": REDACTED,
        }

    def test_redacts_env_dict_values_under_secret_keys(self):
        out = redact_config({"verifier": {"env": {"AWS_BEARER_TOKEN_BEDROCK": "t", "FOO": "bar"}}})
        assert out == {"verifier": {"env": {"AWS_BEARER_TOKEN_BEDROCK": REDACTED, "FOO": "bar"}}}

    @pytest.mark.parametrize(
        "key", ["author", "authors", "AUTHOR", "AUTHORS", "co_author", "authored_cpus", "Authored"]
    )
    def test_keeps_author_metadata_fields(self, key):
        # "AUTHOR"/"AUTHORS"/"AUTHORED" contain "AUTH" but are metadata, not secrets.
        assert redact_config({key: "y"}) == {key: "y"}

    def test_keeps_author_list_metadata(self):
        config = {"authors": [{"name": "n", "email": "e"}]}
        assert redact_config(config) == config

    @pytest.mark.parametrize(
        "key", ["authorization", "authorship", "author_token", "authorize_url", "authoredkey"]
    )
    def test_author_exemption_still_masks_secret_fragments(self, key):
        assert redact_config({key: "CANARY-AUTHOR"}) == {key: REDACTED}

    @pytest.mark.parametrize("assignment", _SECRET_KEY_SCALARS + _OVER_MASKED)
    def test_secret_keys_mask_typed_numeric_and_bool_values(self, assignment):
        # Same cases as argv, as typed JSON values (int/bool) and as strings.
        key, _, raw = assignment.partition("=")
        typed = json.loads(raw)
        assert redact_config({key: typed}) == {key: REDACTED}
        assert redact_config({key: raw}) == {key: REDACTED}
        assert redact_config({"args": [assignment]}) == {"args": [f"{key}={REDACTED}"]}

    def test_secret_keys_mask_float_values(self):
        out = redact_config({"api_key": 1.5, "max_output_tokens": 1.5})
        assert out == {"api_key": REDACTED, "max_output_tokens": REDACTED}

    def test_secret_key_masks_container_values(self):
        out = redact_config({"api_key": {"v": "CANARY"}, "tokens": ["CANARY"]})
        assert out == {"api_key": REDACTED, "tokens": REDACTED}

    def test_secret_key_keeps_none(self):
        # ``None`` carries no secret; keeping it shows the value was never set.
        assert redact_config({"auth": None, "api_key": None}) == {"auth": None, "api_key": None}

    def test_secret_key_keeps_defaultless_template_string(self):
        out = redact_config({"env": {"MY_TOKEN": "${MY_TOKEN}", "X_TOKEN": "${M:-CANARY}"}})
        assert out == {"env": {"MY_TOKEN": "${MY_TOKEN}", "X_TOKEN": "${M:-***}"}}

    def test_string_assignment_in_list_is_masked(self):
        # MCP server ``args: ["api_key=..."]`` and similar list-of-string values.
        config = {
            "mcp_servers": [
                {"command": "npx", "args": ["server", "api_key=CANARY-ARG", "region=us-east-1"]}
            ]
        }
        assert redact_config(config) == {
            "mcp_servers": [
                {"command": "npx", "args": ["server", f"api_key={REDACTED}", "region=us-east-1"]}
            ]
        }

    def test_string_assignment_under_benign_key_is_masked(self):
        out = redact_config({"note": "X_TOKEN=${MISSING:-CANARY-STR}", "plain": "a=b"})
        assert out == {"note": "X_TOKEN=${MISSING:-***}", "plain": "a=b"}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("api_key=sk-CANARY", f"api_key={REDACTED}"),
            ("X_TOKEN=${MISSING:-CANARY}", "X_TOKEN=${MISSING:-***}"),
            ("X_TOKEN=${X_TOKEN}", "X_TOKEN=${X_TOKEN}"),
            ("mean:api_key=CANARY", f"mean:api_key={REDACTED}"),
            ("api_key=multi\nline CANARY", f"api_key={REDACTED}"),
            ("api_key=has spaces CANARY", f"api_key={REDACTED}"),
            ("api_key=https://CANARY@h/x", f"api_key={REDACTED}"),
            ("MY_TOKEN=", f"MY_TOKEN={REDACTED}"),
        ],
    )
    def test_whole_string_assignment_is_masked(self, value, expected):
        assert redact_config({"args": [value]}) == {"args": [expected]}
        assert redact_json(value) == expected

    @pytest.mark.parametrize(
        "prose",
        [
            "aws ec2 create-tags --resources i-123 --tags Key=Name,Value=web-1",
            "found 3 instances; keypair=bench-key region=us-east-1",
            "ThrottlingException: Rate exceeded for key=DescribeInstances",
            "my api_key=internal-space",
            "   =blank-key",
            "=no-key",
            "no assignment here",
        ],
    )
    def test_non_assignment_strings_are_unchanged(self, prose):
        # Only a string that IS ``KEY=VALUE`` (key: no whitespace, no ``=``) is masked.
        assert redact_config({"msg": prose, "args": [prose]}) == {"msg": prose, "args": [prose]}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("API_KEY = sk-CANARY", f"API_KEY ={REDACTED}"),
            (" api_key=sk-CANARY", f" api_key={REDACTED}"),
            ("api_key\n=CANARY", f"api_key\n={REDACTED}"),
        ],
    )
    def test_padded_assignment_is_masked(self, value, expected):
        # The key is stripped before the whitespace test; its text is kept verbatim.
        assert redact_config({"msg": value, "args": [value]}) == {
            "msg": expected,
            "args": [expected],
        }

    def test_bare_tag_token_is_masked_fail_safe(self):
        # The docstring example: a bare token loses everything after ``Key=``.
        assert redact_config({"args": ["Key=Name,Value=web-1"]}) == {"args": [f"Key={REDACTED}"]}

    def test_tags_flag_inside_prose_stays(self):
        prose = "aws ec2 create-tags --resources i-1 --tags Key=Name,Value=web-1"
        assert redact_config({"args": [prose]}) == {"args": [prose]}

    def test_top_level_list_is_scrubbed(self):
        assert redact_json([{"api_key": "CANARY"}, "TOKEN=CANARY", 3]) == [
            {"api_key": REDACTED},
            f"TOKEN={REDACTED}",
            3,
        ]


class TestKnownGaps:
    """Pin the documented gaps (module docstring, "Known gaps"): these pass through.

    A failure here means a gap was closed: update the docstring and move the case
    to the masked tests.
    """

    def test_json_in_string_under_benign_key_passes(self):
        token = 'CONFIG_JSON={"api_token":"CANARY-GAP-JSON"}'
        assert redact_argv(["aws-bench", "run", "--ve", token])[-1] == token
        assert redact_config({"env": {"CONFIG_JSON": token}}) == {"env": {"CONFIG_JSON": token}}

    @pytest.mark.parametrize(
        "url",
        [
            "https://user:CANARY-GAP-USERINFO@mcp.example/sse",
            "https://mcp.example/sse?id=1&api_key=CANARY-GAP-QUERY",
            "https://mcp.example/sse?api_key=CANARY-GAP-QUERY&id=1",
        ],
        ids=["userinfo", "query-secret-last", "query-secret-first"],
    )
    def test_url_credentials_pass_in_config(self, url):
        # In a config string, a ``://`` before the first ``=`` makes it a URL, not
        # an assignment: recorded whole, in either query order.
        assert redact_config({"url": url, "args": [url]}) == {"url": url, "args": [url]}
        assert redact_json(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://user:CANARY-GAP-USERINFO@mcp.example/sse",
            "https://mcp.example/sse?id=1&api_key=CANARY-GAP-QUERY",
        ],
        ids=["userinfo", "query-secret-last"],
    )
    def test_url_credentials_pass_in_argv(self, url):
        # argv decides by the text before the first ``=`` only: no ``=`` (userinfo)
        # or a benign first query key leaves the URL whole.
        argv = ["aws-bench", "run", "--mcp-url", url, f"--mcp-url={url}", url]
        assert redact_argv(argv) == argv

    def test_url_under_benign_assignment_key_passes_in_argv(self):
        # ``U`` decides; the secret-first URL in its value is not inspected.
        token = "U=https://mcp.example/sse?api_key=CANARY-GAP-QUERY&id=1"
        assert redact_argv(["aws-bench", "run", "--ve", token])[-1] == token

    def test_two_token_flag_inside_list_args_passes(self):
        config = {"args": ["server", "--api-key", "CANARY-GAP-ARGS"]}
        assert redact_config(config) == config

    def test_later_metric_kwarg_passes(self):
        # Only the first ``k=v`` of ``--metric type:k=v,k2=v2`` is inspected.
        argv = ["aws-bench", "run", "--metric", "mean:n=1,api_key=CANARY-GAP-METRIC"]
        assert redact_argv(argv) == argv


class TestRecordEnv:
    def test_keeps_allowlisted_keys_verbatim(self):
        env = {
            "AWS_PROFILE": "bench",
            "AWS_DEFAULT_PROFILE": "bench-default",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-west-2",
        }
        assert record_env(env) == env

    def test_drops_unrelated_keys(self):
        out = record_env({"HOME": "/home/x", "SHELL": "/bin/zsh", "AWS_REGION": "us-east-1"})
        assert out == {"AWS_REGION": "us-east-1"}

    def test_drops_aws_credentials(self):
        # Credentials are not on the allowlist, so they never reach disk.
        out = record_env(
            {
                "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "s",
                "AWS_SESSION_TOKEN": "x",
                "AWS_PROFILE": "bench",
            }
        )
        assert out == {"AWS_PROFILE": "bench"}

    def test_drops_other_secret_named_keys(self):
        out = record_env({"GH_TOKEN": "ghp_realtokenvalue", "DB_PASSWORD": "p"})
        assert out == {}

    def test_empty_when_no_allowlisted_keys_present(self):
        assert record_env({"PATH": "/usr/bin", "TERM": "xterm"}) == {}
