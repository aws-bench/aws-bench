"""Secret-safe capture for the command ledger.

- ``record_env`` keeps only an exact allowlist of non-secret AWS resolution
  keys; every other environment variable (including any credential) is dropped.
- ``redact_argv`` / ``redact_config`` / ``redact_json`` are name-driven: they
  mask the value of any flag, ``KEY=VALUE`` token or string, or dict field
  whose name is a secret key (below). A value's content never decides whether
  it is masked; it only selects the masked form of an env template.

Masking rules:

- **Secret key.** A name is secret when, case-insensitively and after blanking
  the words ``AUTHOR``/``AUTHORS``/``AUTHORED`` (metadata, not credentials), it
  contains any of ``SECRET``, ``TOKEN``, ``PASSWORD``, ``KEY``, ``CREDENTIAL``,
  ``AUTH``, ``BEARER`` anywhere. Every value under a secret key is masked,
  whatever its type (numbers and booleans included), except ``None`` and the
  env templates below.
- **Over-masking trade-off.** Substring matching is deliberately broad
  (fail-safe), so benign names that contain a fragment are masked too:
  ``max_tokens=4096`` -> ``max_tokens=***``, ``monkey``, ``KEYBOARD_LAYOUT``.
  No value-based exemption is made, because one cannot tell a count from an
  all-digit credential (``access_tokens=1234567890``, ``API_KEY=12345``).
- **argv.** A secret-named flag masks its next token (``--api-token ***``) or
  its whole inline value (``--api-token=***``). Every other token containing
  ``=`` - the operand of ``--ve``/``--ae``/``--ak``/``--verifier-kwarg``/... or a
  bare positional - and the value of every inline ``--flag=KEY=VALUE`` is
  masked by key, keeping the key visible (``--ak=api_key=***``). Only the text
  before the first ``=`` is the key; it is kept verbatim, padding included
  (``API_KEY = sk-x`` -> ``API_KEY =***``). A token whose key part is strictly
  empty (``=KEY=value``) is masked whole (``***``); no valid invocation
  produces one. A blank key part (``   =value``) is not a secret name, so the
  token is kept. URLs get no special case (fail-safe): in
  ``https://h/p?api_key=sk-x`` the key part is ``https://h/p?api_key``, so the
  token is recorded as ``https://h/p?api_key=***``, truncating the URL.
- **Env templates.** Under a secret key, a defaultless ``${VAR}`` string is
  kept verbatim (it carries no value; Harbor resolves it from the host env). In
  ``${VAR:-default}`` the default is a literal Harbor falls back to, so it is
  masked as ``${VAR:-***}``. Anything else that merely looks like a template
  is a literal and fully masked.
- **Config / JSON.** Dict values under secret keys are masked as above; ``None``
  under a secret key is kept (it shows the value was never set). Values under
  other keys are recursed. A string (dict value or list item) is masked only
  when the whole string is an assignment: the text before its first ``=``,
  stripped of surrounding whitespace, is non-empty and has no internal
  whitespace (``api_key=sk-x`` -> ``api_key=***``, covering MCP server
  ``args: ["api_key=..."]``). Its key decides and is kept verbatim, exactly as
  in argv (``API_KEY = sk-x`` -> ``API_KEY =***``, `` api_key=x`` ->
  `` api_key=***``). That is fail-safe for a bare token: ``Key=Name,Value=web-1``
  is recorded as ``Key=***``. Prose with internal whitespace before its first
  ``=`` (``aws ec2 create-tags --tags Key=Name,Value=web-1``,
  ``found 3; keypair=x``), a string whose key part is empty or blank (``=x``)
  and a URL (a key part containing ``://``; unlike argv) are left unchanged.

Known gaps (pinned by ``TestKnownGaps``; these pass through unmasked):

- Secrets nested in a JSON string under a benign key, e.g.
  ``--ve 'CONFIG_JSON={"api_token": "..."}'``.
- Credentials in URL userinfo or query strings. In config strings, a URL is
  recorded unchanged in either query order (an MCP ``url`` of
  ``https://user:pass@host/sse``, ``...?id=1&api_key=...`` or
  ``...?api_key=...&id=1``). In argv only the text before the first ``=``
  decides, so ``...?api_key=...&id=1`` is masked (see argv above) but userinfo,
  ``...?id=1&api_key=...`` and a URL under a benign assignment key
  (``--ve U=https://...``) pass. A URL is always masked as the value of a
  secret-named key or flag (``api_key=https://...``, ``--api-key https://...``).
- A two-token ``--api-key <value>`` inside list values such as MCP ``args``.
- Any ``k=v`` after the first in ``--metric type:k=v,k2=v2``.

Scope: ``command.json`` (argv and resolved config) only. The sibling
``run.log`` is verbatim DEBUG and outside redaction scope by design. The
snapshot is not redacted: ``ledger`` leaves out the config files but archives
every other job-dir member (including ``result.json``) as-is, and the
Harbor-owned job directory on disk holds everything in cleartext; see the
``ledger`` module docstring. The tree is owner-only (``0o700``/``0o600``);
treat ``~/.aws-bench/logs/`` as sensitive.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

REDACTED = "***"

# The exact env keys recorded in command.json. Everything else — ambient shell
# vars and any AWS credentials — is dropped; only these non-secret AWS
# resolution keys are kept verbatim.
_ENV_ALLOWLIST = frozenset(
    {
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    }
)

# Key-name fragments that mark a value as secret (matched case-insensitively).
# A superset of the fragments in Harbor's sensitive-key pattern, plus BEARER,
# with an explicit AUTHOR-family carve-out (_AUTHOR_WORD) that Harbor lacks:
# Harbor masks e.g. ``authors``, this predicate does not.
_SECRET_KEY_PARTS = ("SECRET", "TOKEN", "PASSWORD", "KEY", "CREDENTIAL", "AUTH", "BEARER")

# "AUTHOR"/"AUTHORS"/"AUTHORED" as a whole word is metadata, not a secret, so
# it is blanked before matching. The trailing letter guard (case-insensitive)
# keeps AUTHORIZATION, AUTHORIZE, AUTHORSHIP, ... secret.
_AUTHOR_WORD = re.compile(r"AUTHOR(S|ED)?(?![A-Z])", re.IGNORECASE)

# Env templates: defaultless ``${VAR}`` vs ``${VAR:-default}``.
_TEMPLATE_PLAIN = re.compile(r"\$\{[^}:]+\}")
_TEMPLATE_DEFAULT = re.compile(r"\$\{([^}:]+):-(.*)\}", re.DOTALL)

# Internal whitespace in a (stripped) key part marks a string as prose.
_WHITESPACE = re.compile(r"\s")


def _is_secret_key(key: str) -> bool:
    upper = _AUTHOR_WORD.sub(" ", key).upper()
    return any(part in upper for part in _SECRET_KEY_PARTS)


def _mask_value(value: str) -> str:
    """The recorded form of a string ``value`` under a secret key."""
    if _TEMPLATE_PLAIN.fullmatch(value):
        return value
    template = _TEMPLATE_DEFAULT.fullmatch(value)
    if template:
        return f"${{{template.group(1)}:-{REDACTED}}}"
    return REDACTED


def _redact_assignment(token: str) -> str:
    """Mask the value of a secret-named ``KEY=VALUE`` token; keep the key visible."""
    key, sep, value = token.partition("=")
    if not sep:
        return token
    if not key:
        # ``=...``: an already-invalid operand; mask it whole for safety.
        return REDACTED
    if not _is_secret_key(key):
        return token
    return f"{key}={_mask_value(value)}"


def redact_argv(argv: list[str]) -> list[str]:
    """Redact secret-named flag values and ``KEY=VALUE`` values; pass the rest through."""
    out: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next:
            out.append(REDACTED)
            redact_next = False
            continue
        if token.startswith("-") and "=" in token:
            # Inline ``--flag=value``: decided by the flag and the value's own
            # ``KEY=`` prefix, never the next token.
            flag, _, value = token.partition("=")
            if _is_secret_key(flag.lstrip("-").replace("-", "_")):
                out.append(f"{flag}={REDACTED}")
            else:
                out.append(f"{flag}={_redact_assignment(value)}")
            continue
        if token.startswith("-") and _is_secret_key(token.lstrip("-").replace("-", "_")):
            out.append(token)
            redact_next = True
            continue
        # Covers spaced operands (``--ve KEY=VALUE``, ``--ak k=v``) and any bare ``KEY=VALUE``.
        out.append(_redact_assignment(token))
    return out


def record_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Record only the exact-allowlisted env keys; drop everything else."""
    return {key: environ[key] for key in _ENV_ALLOWLIST if key in environ}


def _scrub_field(key: str, value: object) -> object:
    if not _is_secret_key(key):
        return redact_json(value)
    if value is None:
        # Carries no secret; keeping it shows the value was never set.
        return None
    if isinstance(value, str):
        return _mask_value(value)
    return REDACTED


def _redact_string(value: str) -> str:
    """Mask a string that IS a secret ``KEY=VALUE``; leave prose and URLs unchanged."""
    key, sep, _ = value.partition("=")
    key = key.strip()
    if not sep or not key or _WHITESPACE.search(key) or "://" in key:
        # Not an assignment: prose, an empty key, or a URL (a known gap).
        return value
    return _redact_assignment(value)


def redact_json(obj: object) -> object:
    """Recursively redact a JSON-shaped value (see the module docstring rules)."""
    if isinstance(obj, dict):
        return {k: _scrub_field(str(k), v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_json(v) for v in obj]
    if isinstance(obj, str):
        return _redact_string(obj)
    return obj


def redact_config(config: dict | None) -> dict | None:
    """Recursively redact a config dict; ``None`` passes through."""
    if config is None:
        return None
    return cast("dict", redact_json(config))
