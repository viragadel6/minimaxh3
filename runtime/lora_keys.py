"""Pure parsing helpers for LoRA specifications and state-dict keys.

This module intentionally has no torch dependency so the parsing rules can be unit
tested (and reused) without importing the deep-learning stack.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from schemas import LoraSpec


_PREFIXES = (
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "model.",
    "lora_unet_",
    "lora_transformer_",
)

#: ``(pattern, style)`` pairs; ``style`` only documents the dialect.
_LORA_KEY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"^(?P<base>.*?)\.lora_(?P<role>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
        ),
        "peft",
    ),
    (
        re.compile(
            r"^lora_(?P<role>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
        ),
        "peft-root",
    ),
    (
        re.compile(
            r"^(?P<base>.*?)\.lora_(?P<role>down|up)(?:\.(?P<adapter>[^.]+))?\.weight$"
        ),
        "kohya",
    ),
    (
        re.compile(r"^lora_(?P<role>down|up)\.weight$"),
        "kohya-root",
    ),
    (
        re.compile(r"^(?P<base>.*?)\.lora\.(?P<role>down|up)\.weight$"),
        "diffusers",
    ),
    (
        re.compile(r"^(?P<base>.*?)_lora\.(?P<role>down|up)\.weight$"),
        "diffusers-suffix",
    ),
    (
        re.compile(r"^(?P<base>.*?)\.lora_linear_layer\.(?P<role>down|up)\.weight$"),
        "peft-linear-layer",
    ),
)

_ALPHA_RE = re.compile(r"^(?P<base>.*?)\.(?:lora_)?alpha$")
_LORA_SUFFIXES = (".lora_linear_layer", ".lora", "_lora")

def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None
    for char in text:
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            buffer.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == separator and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)
    parts.append("".join(buffer))
    return [part.strip() for part in parts if part.strip()]


def _split_outside_brackets(text: str, separator: str) -> tuple[str | None, str | None]:
    """Split on ``separator`` ignoring occurrences inside ``[...]`` (IPv6 hosts)."""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(separator, index):
            return text[:index], text[index + len(separator) :]
        index += 1
    return None, None


def _scheme(value: str) -> str:
    head, sep, _ = value.partition(":")
    return head.strip().lower() if sep else ""


def _parse_entry(entry: str) -> LoraSpec:
    entry = entry.strip()
    if not entry:
        raise ValueError("empty LoRA specification")
    scheme = _scheme(entry)
    source = entry
    weight_name: str | None = None
    scale = 1.0

    head, rest = _split_outside_brackets(entry, "::")
    if head is not None:
        source = head.strip()
        rest = rest.strip()
        if rest and ":" in rest:
            candidate_name, _, candidate_scale = rest.rpartition(":")
            candidate_name = candidate_name.strip()
            candidate_scale = candidate_scale.strip()
            try:
                scale = float(candidate_scale)
            except ValueError as exc:
                raise ValueError(
                    f"invalid LoRA scale {candidate_scale!r} in {entry!r}"
                ) from exc
            weight_name = candidate_name or None
        elif rest:
            try:
                scale = float(rest)
            except ValueError:
                weight_name = rest
            else:
                # ``url::0.8`` is a scale, not a weight file named "0.8".
                weight_name = None
    elif scheme not in ("http", "https", "hf", "data"):
        prefix, _, suffix = entry.rpartition(":")
        if prefix and suffix and "/" not in suffix:
            try:
                scale = float(suffix)
            except ValueError:
                pass
            else:
                source = prefix.strip()

    source = source.strip()
    if not source:
        raise ValueError(f"LoRA specification {entry!r} has no source")
    return LoraSpec(source=source, weight_name=weight_name, scale=scale)


def parse_fused_loras(raw: str) -> list[LoraSpec]:
    """Parse the ``FUSED_LORAS`` setting (JSON list or comma separated shorthand)."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("FUSED_LORAS JSON must be a list")
        specs: list[LoraSpec] = []
        for item in data:
            if isinstance(item, str):
                specs.append(_parse_entry(item))
            elif isinstance(item, dict):
                specs.append(LoraSpec.model_validate(item))
            else:
                raise ValueError(f"unsupported FUSED_LORAS entry: {item!r}")
        return specs
    if _scheme(text) == "data":
        # A data URI contains commas; it can only be expressed through JSON.
        return [_parse_entry(text)]
    return [_parse_entry(entry) for entry in _split_top_level(text)]


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _normalize_role(role: str) -> str:
    return "A" if role in ("A", "down") else "B"


def _match_lora_key(key: str) -> tuple[str, str, str] | None:
    """Split a state-dict key into ``(base_module, role, adapter_name)``.

    The root-module patterns have no ``base``/``adapter`` group, so every group is
    read through ``groupdict`` instead of ``group`` (``group("base")`` raises
    ``IndexError`` for a pattern that does not define it).
    """
    for pattern, _style in _LORA_KEY_PATTERNS:
        match = pattern.match(key)
        if match is None:
            continue
        groups = match.groupdict()
        role = groups.get("role")
        if role is None:  # pragma: no cover - every pattern defines a role
            continue
        return groups.get("base") or "", _normalize_role(role), groups.get("adapter") or "default"
    return None


def _alpha_base(key: str) -> str | None:
    match = _ALPHA_RE.match(key)
    if not match:
        return None
    base = match.group("base") or ""
    for suffix in _LORA_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return _strip_prefixes(base)


def _build_name_maps(name_list: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    underscore: dict[str, str] = {}
    ambiguous: set[str] = set()
    for mod_name in name_list:
        stripped = _strip_prefixes(mod_name)
        for alias in (mod_name, stripped):
            existing = exact.get(alias)
            if existing is not None and existing != mod_name:
                ambiguous.add(alias)
            exact[alias] = mod_name
        forms = {
            stripped.replace(".", "_"),
            f"lora_unet_{stripped.replace('.', '_')}",
            f"lora_transformer_{stripped.replace('.', '_')}",
        }
        for form in forms:
            existing = underscore.get(form)
            if existing is not None and existing != mod_name:
                ambiguous.add(form)
            underscore[form] = mod_name
    for alias in ambiguous:
        exact.pop(alias, None)
        underscore.pop(alias, None)
    return exact, underscore


def _resolve_module_name(
    base: str,
    exact_map: dict[str, str],
    underscore_map: dict[str, str],
) -> str | None:
    trimmed = base.removeprefix("lora_unet_").removeprefix("lora_transformer_")
    candidates = (base, trimmed, trimmed.replace(".", "_"))
    for candidate in candidates:
        if candidate in exact_map:
            return exact_map[candidate]
        if candidate in underscore_map:
            return underscore_map[candidate]
    return None


