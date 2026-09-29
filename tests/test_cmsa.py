from __future__ import annotations

from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest

from deepinterestgr import CLIP_MODEL_ID, CMSAFusion


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


class FakeTensor:
    def __init__(self, values, *, batch_size=1):
        self.values = tuple(float(value) for value in values)
        self.shape = (batch_size, len(self.values))


class FakeVision:
    def __init__(self, values=(1.0, 2.0), *, batch_size=1) -> None:
        self.training = True
        self.parameter = FakeParameter()
        self.values = values
        self.batch_size = batch_size

    def parameters(self):
        return [self.parameter]

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, image_inputs):
        return SimpleNamespace(
            pooler_output=FakeTensor(self.values, batch_size=self.batch_size)
        )


class FakeLinear:
    def __init__(self, in_features, out_features, *, bias):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.training = True
        self.parameter = FakeParameter()

    def parameters(self):
        return [self.parameter]

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, value):
        if len(value.values) != self.in_features:
            raise AssertionError("wrong linear input dimension")
        total = sum(value.values)
        return FakeTensor(total + index for index in range(self.out_features))


class FakeOps:
    def no_grad(self):
        return nullcontext()

    def concatenate(self, values):
        return FakeTensor((*values[0].values, *values[1].values))

    def all_finite(self, value):
        return all(math.isfinite(element) for element in value.values)


class CMSATests(unittest.TestCase):
    def make_fusion(self, vision=None):
        created = []

        def factory(in_features, out_features, *, bias):
            layer = FakeLinear(in_features, out_features, bias=bias)
            created.append(layer)
            return layer

        fusion = CMSAFusion(
            vision_model=vision or FakeVision(),
            vision_dim=2,
            text_dim=3,
            rqvae_dim=4,
            linear_factory=factory,
            tensor_ops=FakeOps(),
        )
        return fusion, created

    def test_reported_model_and_direct_linear_shapes(self):
        fusion, created = self.make_fusion()
        self.assertEqual(CLIP_MODEL_ID, "openai/clip-vit-large-patch14")
        self.assertEqual(
            [(layer.in_features, layer.out_features, layer.bias) for layer in created],
            [(2, 3, False), (6, 4, False)],
        )
        output = fusion(FakeTensor((5.0, 6.0, 7.0)), "image")
        self.assertEqual(output.shape, (1, 4))
        self.assertFalse(hasattr(fusion, "activation"))

    def test_clip_is_frozen_and_stays_eval_after_train(self):
        vision = FakeVision()
        fusion, created = self.make_fusion(vision)
        self.assertFalse(vision.parameter.requires_grad)
        self.assertFalse(vision.training)
        self.assertTrue(all(layer.parameter.requires_grad for layer in created))
        fusion.train(True)
        self.assertFalse(vision.training)
        self.assertTrue(all(layer.training for layer in created))
        self.assertEqual(list(fusion.trainable_parameters()), [
            created[0].parameter,
            created[1].parameter,
        ])

    def test_nonfinite_input_and_vision_feature_are_rejected(self):
        fusion, _ = self.make_fusion()
        with self.assertRaisesRegex(ValueError, "text_embedding.*finite"):
            fusion(FakeTensor((1.0, float("nan"), 2.0)), "image")
        fusion, _ = self.make_fusion(FakeVision(values=(1.0, float("inf"))))
        with self.assertRaisesRegex(ValueError, "vision feature.*finite"):
            fusion(FakeTensor((1.0, 2.0, 3.0)), "image")

    def test_dimensions_are_explicit_and_checked(self):
        with self.assertRaisesRegex(ValueError, "text_dim"):
            CMSAFusion(
                vision_model=FakeVision(),
                vision_dim=2,
                text_dim=0,
                rqvae_dim=4,
                linear_factory=FakeLinear,
                tensor_ops=FakeOps(),
            )
        fusion, _ = self.make_fusion()
        with self.assertRaisesRegex(ValueError, "text_dim=3"):
            fusion(FakeTensor((1.0, 2.0)), "image")
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            fusion, _ = self.make_fusion(FakeVision(batch_size=2))
            fusion(FakeTensor((1.0, 2.0, 3.0), batch_size=1), "image")


if __name__ == "__main__":
    unittest.main()
