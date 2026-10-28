import pytest

from runtime import lora_keys
from schemas import LoraSpec


# ---------------------------------------------------------------------------
# entry splitting
# ---------------------------------------------------------------------------
def test_split_top_level_respects_brackets_and_quotes():
    assert lora_keys._split_top_level("a,b,c") == ["a", "b", "c"]
    assert lora_keys._split_top_level("a(b,c),d") == ["a(b,c)", "d"]
    assert lora_keys._split_top_level('"x,y",z') == ['"x,y"', "z"]
    assert lora_keys._split_top_level("a,,b") == ["a", "b"]


def test_double_colon_split_ignores_ipv6_brackets():
    head, rest = lora_keys._split_outside_brackets("http://[::1]/a.safetensors::w", "::")
    assert head == "http://[::1]/a.safetensors"
    assert rest == "w"


# ---------------------------------------------------------------------------
# fused LoRA specs
# ---------------------------------------------------------------------------
def test_parse_bare_source():
    specs = lora_keys.parse_fused_loras("adapter.safetensors")
    assert specs == [LoraSpec(source="adapter.safetensors", weight_name=None, scale=1.0)]


def test_parse_weight_and_scale():
    specs = lora_keys.parse_fused_loras("repo/id::weights.safetensors:0.75")
    assert specs[0].source == "repo/id"
    assert specs[0].weight_name == "weights.safetensors"
    assert specs[0].scale == pytest.approx(0.75)


def test_parse_empty_weight_is_none():
    assert lora_keys.parse_fused_loras("repo/id::").weight_name if False else True
    specs = lora_keys.parse_fused_loras("repo/id::")
    assert specs[0].weight_name is None


def test_parse_url_keeps_port():
    specs = lora_keys.parse_fused_loras("https://host:8080/a.safetensors")
    assert specs[0].source == "https://host:8080/a.safetensors"
    assert specs[0].scale == 1.0


def test_parse_url_with_scale_suffix():
    specs = lora_keys.parse_fused_loras("https://host/a.safetensors::0.5")
    assert specs[0].scale == pytest.approx(0.5)
    assert specs[0].weight_name is None


def test_parse_local_path_with_scale():
    specs = lora_keys.parse_fused_loras("/models/a.safetensors:0.25")
    assert specs[0].source == "/models/a.safetensors"
    assert specs[0].scale == pytest.approx(0.25)


def test_parse_json_forms():
    specs = lora_keys.parse_fused_loras(
        '[{"source": "a", "scale": 0.5}, "b::c:0.25"]'
    )
    assert [s.source for s in specs] == ["a", "b"]
    assert [s.scale for s in specs] == [0.5, 0.25]


def test_parse_empty_input():
    assert lora_keys.parse_fused_loras("") == []
    assert lora_keys.parse_fused_loras("  ") == []


def test_parse_keeps_data_uri_as_source():
    # Parsing preserves the source; ``resolve_lora`` is what refuses data URIs.
    specs = lora_keys.parse_fused_loras("data:application/octet-stream;base64,AAAA")
    assert specs[0].source.startswith("data:")


def test_parse_rejects_bad_scale():
    with pytest.raises(ValueError):
        lora_keys.parse_fused_loras("a::b:not-a-number")
    with pytest.raises(ValueError):
        lora_keys.parse_fused_loras("a::b:-1")


def test_parse_rejects_empty_source():
    with pytest.raises(ValueError):
        lora_keys.parse_fused_loras("::weight")


def test_parse_multiple_entries():
    specs = lora_keys.parse_fused_loras("a,b::c,d:2")
    assert len(specs) == 3
    assert specs[2].scale == 2.0


# ---------------------------------------------------------------------------
# state-dict keys
# ---------------------------------------------------------------------------
def test_prefix_stripping():
    assert lora_keys._strip_prefixes("base_model.model.x.lora") == "x.lora"
    assert lora_keys._strip_prefixes("model.diffusion_model.y") == "y"
    assert lora_keys._strip_prefixes("keep.me") == "keep.me"


@pytest.mark.parametrize(
    "key,expected",
    [
        ("blocks.0.attn.lora_A.weight", ("blocks.0.attn", "A", "default")),
        ("blocks.0.attn.lora_B.weight", ("blocks.0.attn", "B", "default")),
        (
            "blocks.0.attn.lora_A.my_adapter.weight",
            ("blocks.0.attn", "A", "my_adapter"),
        ),
        ("blocks.0.attn.lora_down.weight", ("blocks.0.attn", "A", "default")),
        ("blocks.0.attn.lora_up.weight", ("blocks.0.attn", "B", "default")),
        ("blocks.0.attn.lora.down.weight", ("blocks.0.attn", "A", "default")),
        ("blocks.0.attn_lora.down.weight", ("blocks.0.attn", "A", "default")),
        ("lora_A.weight", ("", "A", "default")),
        ("lora_down.weight", ("", "A", "default")),
    ],
)
def test_match_lora_key(key, expected):
    assert lora_keys._match_lora_key(key) == expected


def test_match_rejects_unrelated_keys():
    assert lora_keys._match_lora_key("blocks.0.attn.to_q.weight") is None
    assert lora_keys._match_lora_key("blocks.0.attn.lora_X.weight") is None


@pytest.mark.parametrize(
    "key,expected",
    [
        ("blocks.0.attn.alpha", "blocks.0.attn"),
        ("blocks.0.attn.lora_alpha", "blocks.0.attn"),
        ("blocks.0.attn.lora.alpha", "blocks.0.attn"),
        ("blocks.0.attn_lora.alpha", "blocks.0.attn"),
    ],
)
def test_alpha_base(key, expected):
    assert lora_keys._alpha_base(key) == expected


def test_alpha_base_rejects_weight_keys():
    assert lora_keys._alpha_base("blocks.0.attn.weight") is None


def test_name_maps_detect_ambiguity():
    exact, underscore = lora_keys._build_name_maps(
        ["a.b", "a_b", "unambiguous.one"]
    )
    assert exact["a.b"] == "a.b"
    # "a_b" is ambiguous (both a real module name and the underscore form of "a.b").
    assert "a_b" not in exact
    # Two modules collapse onto the same underscore form: neither may be inferred.
    assert underscore.get("a_b") is None
    assert underscore["unambiguous_one"] == "unambiguous.one"


def test_resolve_module_name_prefers_exact():
    assert (
        lora_keys._resolve_module_name("a.b", {"a.b": "a.b"}, {"a_b": "a.b"}) == "a.b"
    )


def test_resolve_module_name_uses_underscore_map():
    assert (
        lora_keys._resolve_module_name("a_b", {"a.b": "a.b"}, {"a_b": "a.b"}) == "a.b"
    )


def test_resolve_module_name_unknown_returns_none():
    assert lora_keys._resolve_module_name("missing", {"a.b": "a.b"}, {}) is None
