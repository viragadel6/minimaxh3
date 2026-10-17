import inspect
from collections.abc import Mapping

from shapes import FPS


def _names(blocks, seen=None):
    seen = set() if seen is None else seen
    if blocks is None or id(blocks) in seen:
        return set()
    seen.add(id(blocks))
    result = set()
    for value in getattr(blocks, "input_names", ()) or ():
        result.add(value if isinstance(value, str) else value.name)
    for attribute in ("sub_blocks", "blocks"):
        children = getattr(blocks, attribute, None)
        if isinstance(children, Mapping):
            children = children.values()
        if isinstance(children, (list, tuple)) or hasattr(children, "__iter__"):
            for child in children:
                if not isinstance(child, str):
                    result.update(_names(child, seen))
    return result


def _component(pipe, name):
    components = getattr(pipe, "components", {})
    if isinstance(components, Mapping) and name in components:
        return components[name]
    return getattr(pipe, name, None)


class PipelineAdapter:
    ALIASES = {
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

    def __init__(self, pipe, blocks_module):
        self.pipe = pipe
        self.blocks_module = blocks_module
        names = _names(getattr(pipe, "blocks", None))
        signature = inspect.signature(pipe.__call__)
        names.update(
            name for name, parameter in signature.parameters.items()
            if parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            )
        )
        self.mapping = {
            standard: next((name for name in aliases if name in names), None)
            for standard, aliases in self.ALIASES.items()
        }
        missing = [
            name for name in (
                "prompt", "height", "width", "num_frames", "steps", "generator"
            )
            if self.mapping[name] is None
        ]
        if missing:
            raise RuntimeError(
                "Unsupported MiniMaxH3 ModularPipeline inputs: " + ", ".join(missing)
            )

    def _reference(self, item):
        kind = item["type"]
        anchor = item.get("frame_index")
        candidates = []
        preferred = (
            f"{kind.title()}Reference",
            "MiniMaxH3Reference",
            "Reference",
            "ReferenceInput",
        )
        for name in preferred:
            factory = getattr(self.blocks_module, name, None)
            if callable(factory) and factory not in candidates:
                candidates.append(factory)
        for name, factory in vars(self.blocks_module).items():
            if "reference" in name.lower() and inspect.isclass(factory):
                if factory not in candidates:
                    candidates.append(factory)

        failures = []
        for factory in candidates:
            try:
                parameters = inspect.signature(factory).parameters
            except (TypeError, ValueError):
                continue

            values = {
                "type": kind,
                "modality": kind,
                "reference_type": kind,
                "data": item["data"],
                "source": item["data"],
                "media": item["data"],
                "reference": item["data"],
                "strength": item.get("strength"),
                "conditioning_scale": item.get("strength"),
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
                "start_time": item.get("start_seconds"),
                "end_time": item.get("end_seconds"),
            }
            values[kind] = item["data"]
            if kind == "video":
                values["frames"] = item["data"]
            if anchor is not None:
                for name in (
                    "frame_index", "frame_idx", "index",
                    "target_frame_index", "target_frame",
                ):
                    values[name] = anchor
                for name in ("frame_indices", "frame_indexes", "indices"):
                    values[name] = [anchor]

            anchor_names = {
                "frame_index", "frame_idx", "index", "target_frame_index",
                "target_frame", "frame_indices", "frame_indexes", "indices",
            }
            data_names = {"data", "source", "media", "reference", kind}
            if kind == "video":
                data_names.add("frames")
            if not data_names.intersection(parameters):
                continue
            if anchor is not None and not anchor_names.intersection(parameters):
                continue
            if item.get("strength") is not None:
                if not {"strength", "conditioning_scale"}.intersection(parameters):
                    continue

            kwargs = {
                name: values[name]
                for name, parameter in parameters.items()
                if name in values
                and values[name] is not None
                and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
            }
            required = {
                name for name, parameter in parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
            if not required.issubset(kwargs):
                continue
            try:
                return factory(**kwargs)
            except (ValueError, TypeError) as exc:
                failures.append(f"{factory.__name__}: {exc}")

        detail = "; ".join(failures[:3])
        raise RuntimeError(
            f"No compatible {kind} reference constructor was exported by "
            f"{self.blocks_module.__name__}. "
            f"An explicit media field and frame-index field are required for anchors. "
            f"{detail}"
        )

    def build_references(self, first_frame, last_frame, references, num_frames):
        items = []
        if first_frame is not None:
            items.append({"type": "image", "data": first_frame, "frame_index": 0})
        items.extend(dict(item) for item in (references or []))
        if last_frame is not None:
            items.append({
                "type": "image",
                "data": last_frame,
                "frame_index": num_frames - 1,
            })
        return [self._reference(item) for item in items]

    def run(
        self,
        *,
        prompt,
        height,
        width,
        num_frames,
        steps,
        generator,
        first_frame=None,
        last_frame=None,
        references=None,
    ):
        conditions = self.build_references(
            first_frame, last_frame, references, num_frames
        )
        values = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "steps": steps,
            "generator": generator,
            "fps": FPS,
            "output_type": "pt",
        }
        kwargs = {
            self.mapping[name]: value
            for name, value in values.items()
            if self.mapping[name] is not None
        }
        if self.mapping["references"] is not None:
            kwargs[self.mapping["references"]] = conditions
        elif conditions:
            raise RuntimeError("The pipeline does not expose a reference input")

        result = self.pipe(**kwargs, output=["videos", "audio"])

        def read(name):
            if isinstance(result, Mapping):
                return result.get(name)
            getter = getattr(result, "get", None)
            if callable(getter):
                try:
                    return getter(name)
                except (KeyError, TypeError):
                    pass
            return getattr(result, name, None)

        if isinstance(result, tuple) and len(result) == 2:
            video, audio = result
        else:
            video, audio = read("videos"), read("audio")
        if video is None:
            raise RuntimeError("The pipeline did not return its 'videos' output")
        if isinstance(video, (list, tuple)) and len(video) == 1:
            video = video[0]
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            audio = audio[0]
        return video, audio

    def sample_rate(self):
        for name in ("audio_vae", "vocoder"):
            component = _component(self.pipe, name)
            if component is None:
                continue
            for source in (component, getattr(component, "config", None)):
                if source is None:
                    continue
                for field in ("sample_rate", "sampling_rate", "audio_sample_rate"):
                    value = (
                        source.get(field)
                        if isinstance(source, Mapping)
                        else getattr(source, field, None)
                    )
                    if value is not None and int(value) >= 8000:
                        return int(value)
        raise RuntimeError("The pipeline does not expose its audio sample rate")
