"""Cross-Modal Semantic Augmentation (CMSA) caption and fusion paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
import math
from typing import Any, Callable, Protocol

from .api import CompletionAdapter


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


try:  # Keep package imports usable when optional ML dependencies are absent.
    from torch.nn import Module as _ModuleBase
except ImportError:
    class _ModuleBase:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.training = True

        def train(self, mode: bool = True) -> "_ModuleBase":
            self.training = bool(mode)
            return self

        def eval(self) -> "_ModuleBase":
            return self.train(False)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)


class TensorOps(Protocol):
    def no_grad(self) -> AbstractContextManager[Any]: ...

    def concatenate(self, values: tuple[Any, Any]) -> Any: ...

    def all_finite(self, value: Any) -> bool: ...


class _TorchTensorOps:
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'ml' dependencies or inject tensor_ops"
            ) from exc
        self.torch = torch

    def no_grad(self) -> AbstractContextManager[Any]:
        return self.torch.no_grad()

    def concatenate(self, values: tuple[Any, Any]) -> Any:
        return self.torch.cat(values, dim=-1)

    def all_finite(self, value: Any) -> bool:
        return bool(self.torch.isfinite(value).all().item())


def _default_linear_factory(in_features: int, out_features: int, *, bias: bool) -> Any:
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional 'ml' dependencies or inject linear_factory"
        ) from exc
    return nn.Linear(in_features, out_features, bias=bias)


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{name} must expose a shape")
    try:
        parsed = tuple(int(dimension) for dimension in shape)
        if not parsed:
            raise ValueError(f"{name} must not be scalar")
        return parsed
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} has an invalid shape") from exc


def _validate_embedding_shape(
    value: Any,
    *,
    name: str,
    feature_dim: int,
    dimension_name: str,
) -> tuple[int, ...]:
    shape = _shape(value, name=name)
    if len(shape) != 2:
        raise ValueError(
            f"{name} must have shape (batch_size, {dimension_name}={feature_dim})"
        )
    if shape[-1] != feature_dim:
        raise ValueError(
            f"{name} last dimension must equal {dimension_name}={feature_dim}"
        )
    return shape


def _set_parameter_trainability(module: Any, trainable: bool) -> None:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return
    for parameter in parameters():
        setter = getattr(parameter, "requires_grad_", None)
        if callable(setter):
            setter(trainable)
        else:
            parameter.requires_grad = trainable


def _module_parameters(module: Any) -> Iterable[Any]:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        yield from parameters()


class CMSACaptioner:
    """Generate one recommendation-oriented caption for exactly one image URL."""

    _PROMPT = (
        "Describe this single item image using only visible evidence relevant to item "
        "recommendation: appearance, material or texture cues, usage setting, and scene. "
        "Do not infer a person's identity, private traits, emotions, or psychology. Return "
        "one concise factual caption and no surrounding prose."
    )

    def __init__(self, adapter: CompletionAdapter) -> None:
        self.adapter = adapter

    def caption(self, image_url: str) -> str:
        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image_url must identify exactly one non-empty URL or data URL")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url.strip()},
                    },
                ],
            }
        ]
        caption = self.adapter.complete(messages, temperature=0.0)
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption API returned empty text")
        return caption.strip()


class CMSAFusion(_ModuleBase):
    """Frozen CLIP vision path with direct trainable linear projections.

    The implemented equations are exactly ``e_vis = W_v(f_vis(image))`` and
    ``e_mm = W_f(concat(e_text, e_vis))``. No activation, normalization, or
    additional projection is applied.
    """

    def __init__(
        self,
        *,
        vision_model: Any,
        vision_dim: int,
        text_dim: int,
        rqvae_dim: int,
        linear_factory: Callable[..., Any] | None = None,
        tensor_ops: TensorOps | None = None,
        vision_feature_getter: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("vision_dim", vision_dim),
            ("text_dim", text_dim),
            ("rqvae_dim", rqvae_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vision_model is None:
            raise ValueError("vision_model is required")

        factory = linear_factory or _default_linear_factory
        self.vision_model = vision_model
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.rqvae_dim = rqvae_dim
        # Bias is disabled so W_v and W_f are the direct matrices in the paper.
        self.W_v = factory(vision_dim, text_dim, bias=False)
        self.W_f = factory(2 * text_dim, rqvae_dim, bias=False)
        self._tensor_ops = tensor_ops or _TorchTensorOps()
        self._vision_feature_getter = (
            vision_feature_getter or self._clip_pooler_output
        )

        _set_parameter_trainability(self.vision_model, False)
        _set_parameter_trainability(self.W_v, True)
        _set_parameter_trainability(self.W_f, True)
        self.vision_model.eval()

    @staticmethod
    def _clip_pooler_output(output: Any) -> Any:
        feature = getattr(output, "pooler_output", None)
        if feature is None:
            raise ValueError("CLIP vision output must expose pooler_output")
        return feature

    @classmethod
    def from_pretrained_clip(
        cls,
        *,
        text_dim: int,
        rqvae_dim: int,
        linear_factory: Callable[..., Any] | None = None,
        tensor_ops: TensorOps | None = None,
    ) -> "CMSAFusion":
        """Load the explicitly reported CLIP vision model on caller request."""

        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'ml' dependencies to load CLIP"
            ) from exc
        vision_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
        # Make the frozen/evaluation contract explicit before constructing projections.
        vision_dim = int(vision_model.config.hidden_size)
        return cls(
            vision_model=vision_model,
            vision_dim=vision_dim,
            text_dim=text_dim,
            rqvae_dim=rqvae_dim,
            linear_factory=linear_factory,
            tensor_ops=tensor_ops,
        )

    def train(self, mode: bool = True) -> "CMSAFusion":
        super().train(mode)
        for projection in (self.W_v, self.W_f):
            train = getattr(projection, "train", None)
            if callable(train):
                train(mode)
        # ``Module.train`` recurses; force the frozen encoder back to eval.
        self.vision_model.eval()
        return self

    def trainable_parameters(self) -> Iterable[Any]:
        """Yield only W_v and W_f parameters for optimizer construction."""

        seen: set[int] = set()
        for module in (self.W_v, self.W_f):
            for parameter in _module_parameters(module):
                marker = id(parameter)
                if marker not in seen and getattr(parameter, "requires_grad", True):
                    seen.add(marker)
                    yield parameter

    def forward(self, text_embedding: Any, image_inputs: Any) -> Any:
        """Fuse batched text embeddings with matching batched CLIP inputs."""

        if not self._tensor_ops.all_finite(text_embedding):
            raise ValueError("text_embedding must contain only finite values")
        text_shape = _validate_embedding_shape(
            text_embedding,
            name="text_embedding",
            feature_dim=self.text_dim,
            dimension_name="text_dim",
        )

        with self._tensor_ops.no_grad():
            if isinstance(image_inputs, Mapping):
                vision_output = self.vision_model(**dict(image_inputs))
            else:
                vision_output = self.vision_model(image_inputs)
            vision_feature = self._vision_feature_getter(vision_output)

        if not self._tensor_ops.all_finite(vision_feature):
            raise ValueError("vision feature must contain only finite values")
        vision_shape = _validate_embedding_shape(
            vision_feature,
            name="vision feature",
            feature_dim=self.vision_dim,
            dimension_name="vision_dim",
        )
        if text_shape[0] != vision_shape[0]:
            raise ValueError("text and vision batch sizes must match")

        visual_embedding = self.W_v(vision_feature)
        if not self._tensor_ops.all_finite(visual_embedding):
            raise ValueError("W_v output must contain only finite values")
        fused = self._tensor_ops.concatenate((text_embedding, visual_embedding))
        multimodal_embedding = self.W_f(fused)
        if not self._tensor_ops.all_finite(multimodal_embedding):
            raise ValueError("W_f output must contain only finite values")
        return multimodal_embedding
