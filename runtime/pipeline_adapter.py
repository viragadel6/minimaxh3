import inspect
from collections.abc import Mapping

from shapes import FPS


def _collect_input_names(block, visited: set[int] | None = None) -> set[str]:
    if visited is None:
        visited = set()
    if block is None or id(block) in visited:
        return set()
    visited.add(id(block))
    names: set[str] = set()
    raw_names = getattr(block, "input_names", None)
    if raw_names:
        for item in raw_names:
            names.add(item if isinstance(item, str) else getattr(item, "name", str(item)))
    for attr in ("sub_blocks", "blocks"):
        children = getattr(block, attr, None)
        if isinstance(children, Mapping):
            children = list(children.values())
        if isinstance(children, (list, tuple, set)):
            for child in children:
                if not isinstance(child, str):
                    names.update(_collect_input_names(child, visited))
    return names


class PipelineAdapter:
    CANONICAL_CANDIDATES: dict[str, tuple[str, ...]] = {
        "prompt": ("prompt",),
        "height": ("height",),
        "width": ("width",),
        "num_frames": ("num_frames",),
        "steps": ("num_inference_steps",),
        "generator": ("generator",),
        "fps": ("frame_rate", "fps"),
        "references": ("references", "conditions"),
        "output_type": ("output_type",),
    }

    def __init__(self, pipe, blocks_module) -> None:
        self.pipe = pipe
        self.blocks_module = blocks_module
        discovered = _collect_input_names(getattr(pipe, "blocks", None))
        try:
            sig = inspect.signature(pipe.__call__)
            for param_name, param in sig.parameters.items():
                if param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    discovered.add(param_name)
        except (TypeError, ValueError):
            pass

        self.arg_map: dict[str, str] = {}
        for canonical, candidates in self.CANONICAL_CANDIDATES.items():
            matched = next((c for c in candidates if c in discovered), None)
            self.arg_map[canonical] = matched if matched is not None else candidates[0]

    def _construct_reference(self, entry: dict):
        ref_type = entry["type"]
        frame_index = entry.get("frame_index")
        data = entry["data"]
        strength = entry.get("strength")
        start_seconds = entry.get("start_seconds")
        end_seconds = entry.get("end_seconds")

        candidates = []
        if self.blocks_module is not None:
            for cls_name in (
                f"{ref_type.capitalize()}Reference",
                "MiniMaxH3Reference",
                "Reference",
                "ReferenceCondition",
            ):
                obj = getattr(self.blocks_module, cls_name, None)
                if callable(obj) and obj not in candidates:
                    candidates.append(obj)
            for attr_name, obj in vars(self.blocks_module).items():
                if "reference" in attr_name.lower() and callable(obj) and obj not in candidates:
                    candidates.append(obj)

        candidate_values = {
            "type": ref_type,
            "modality": ref_type,
            "reference_type": ref_type,
            "data": data,
            "source": data,
            "media": data,
            "reference": data,
            ref_type: data,
            "frames": data if ref_type == "video" else None,
            "strength": strength,
            "scale": strength,
            "conditioning_scale": strength,
            "frame_index": frame_index,
            "frame_idx": frame_index,
            "index": frame_index,
            "target_frame": frame_index,
            "frame_range": (
                (frame_index, frame_index + 1) if frame_index is not None else None
            ),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "start_time": start_seconds,
            "end_time": end_seconds,
        }

        for factory in candidates:
            try:
                sig = inspect.signature(factory)
            except (TypeError, ValueError):
                continue
            kwargs = {}
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            for pname, param in sig.parameters.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if pname in candidate_values and candidate_values[pname] is not None:
                    kwargs[pname] = candidate_values[pname]
            if not kwargs and accepts_var_kw:
                kwargs = {
                    k: v
                    for k, v in (
                        ("type", ref_type),
                        ("data", data),
                        ("strength", strength),
                        ("frame_index", frame_index),
                        ("start_seconds", start_seconds),
                        ("end_seconds", end_seconds),
                    )
                    if v is not None
                }
            if kwargs:
                try:
                    return factory(**kwargs)
                except Exception:
                    continue

        return {
            k: v
            for k, v in (
                ("type", ref_type),
                ("data", data),
                ("strength", strength),
                ("frame_index", frame_index),
                ("start_seconds", start_seconds),
                ("end_seconds", end_seconds),
            )
            if v is not None
        }

    def build_references(
        self,
        first_frame,
        last_frame,
        references: list[dict] | None,
        num_frames: int,
    ) -> list:
        built = []
        if first_frame is not None:
            built.append(
                self._construct_reference(
                    {"type": "image", "data": first_frame, "frame_index": 0}
                )
            )
        for item in references or []:
            built.append(self._construct_reference(dict(item)))
        if last_frame is not None:
            built.append(
                self._construct_reference(
                    {
                        "type": "image",
                        "data": last_frame,
                        "frame_index": num_frames - 1,
                    }
                )
            )
        return built

    def run(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        generator,
        first_frame=None,
        last_frame=None,
        references: list[dict] | None = None,
        fps: int = FPS,
        output_type: str = "pt",
    ):
        built_refs = self.build_references(
            first_frame, last_frame, references, num_frames
        )
        canonical_inputs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "steps": steps,
            "generator": generator,
            "fps": fps,
            "output_type": output_type,
        }
        call_kwargs = {
            self.arg_map[k]: v for k, v in canonical_inputs.items() if k in self.arg_map
        }
        if built_refs:
            call_kwargs[self.arg_map["references"]] = built_refs

        out = self.pipe(**call_kwargs, output=["videos", "audio"])
        if isinstance(out, tuple) and len(out) == 2:
            video, audio = out
        elif isinstance(out, Mapping):
            video = out.get("videos", out.get("video"))
            audio = out.get("audio", out.get("audios"))
        else:
            video = getattr(out, "videos", getattr(out, "video", None))
            audio = getattr(out, "audio", getattr(out, "audios", None))

        if isinstance(video, (list, tuple)) and len(video) == 1:
            video = video[0]
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            audio = audio[0]
        return video, audio

    def sample_rate(self) -> int:
        components = getattr(self.pipe, "components", {}) or {}
        for comp_name in ("audio_vae", "vocoder"):
            comp = (
                components.get(comp_name)
                if isinstance(components, Mapping)
                else getattr(self.pipe, comp_name, None)
            )
            if comp is None:
                comp = getattr(self.pipe, comp_name, None)
            if comp is None:
                continue
            for holder in (comp, getattr(comp, "config", None)):
                if holder is None:
                    continue
                for attr in ("sample_rate", "sampling_rate", "audio_sample_rate"):
                    val = (
                        holder.get(attr)
                        if isinstance(holder, Mapping)
                        else getattr(holder, attr, None)
                    )
                    if val is not None:
                        return int(val)
        return 44100
