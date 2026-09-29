from __future__ import annotations

from types import SimpleNamespace
import builtins
import unittest
from unittest.mock import patch

from deepinterestgr import (
    BinaryLabel,
    CMSACaptioner,
    DCIMExtractor,
    build_enriched_text,
    OpenAICompatibleAdapter,
    QARMJudge,
    SchemaValidationError,
)
from deepinterestgr.schemas import DescriptorResponse


class QueueAdapter:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, *, response_format=None, temperature=0.0):
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
            }
        )
        return self.responses.pop(0)


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" response "))]
        )


class SchemaAndAPITests(unittest.TestCase):
    def test_dcim_keeps_order_and_confidences(self):
        adapter = QueueAdapter(
            '{"descriptors":['
            '{"descriptor":"indoor court practice","confidence":0.9},'
            '{"descriptor":"low-bounce ball control","confidence":0.7}]}'
        )
        response = DCIMExtractor(adapter).extract(
            title="Training ball",
            description="Low-bounce construction",
            caption="Ball on an indoor court",
        )
        self.assertEqual(
            [value.descriptor for value in response.descriptors],
            ["indoor court practice", "low-bounce ball control"],
        )
        call = adapter.calls[0]
        self.assertTrue(call["response_format"]["json_schema"]["strict"])
        prompt_text = str(call["messages"])
        system_prompt = call["messages"][0]["content"].replace("\n", " ")
        self.assertIn("title", prompt_text)
        self.assertIn("description", prompt_text)
        self.assertIn("caption", prompt_text)
        self.assertIn("latent usage or purchase motivations", system_prompt)
        self.assertIn("not a verified psychological state", system_prompt)
        self.assertIn("a personalized claim about a particular user", system_prompt)

    def test_enriched_text_keeps_evidence_and_descriptor_order(self):
        descriptors = DescriptorResponse.from_json(
            '{"descriptors":['
            '{"descriptor":"indoor court practice","confidence":0.9},'
            '{"descriptor":"low-bounce control","confidence":0.7}]}'
        )
        enriched = build_enriched_text(
            title="Training ball",
            description="Low-bounce construction",
            caption="Ball on an indoor court",
            descriptors=descriptors,
        )
        self.assertEqual(
            enriched.splitlines(),
            [
                "Training ball",
                "Low-bounce construction",
                "Ball on an indoor court",
                "indoor court practice",
                "low-bounce control",
            ],
        )
        self.assertNotIn("0.9", enriched)
        self.assertNotIn("0.7", enriched)

    def test_dcim_rejects_extra_fields_without_fallback(self):
        adapter = QueueAdapter(
            '{"descriptors":[{"descriptor":"court play","confidence":0.8,'
            '"explanation":"extra"}]}'
        )
        with self.assertRaisesRegex(SchemaValidationError, "extra"):
            DCIMExtractor(adapter).extract(
                title="Ball", description="Indoor", caption="Court"
            )
        self.assertEqual(adapter.responses, [])

    def test_duplicate_normalized_descriptors_are_rejected(self):
        payload = (
            '{"descriptors":['
            '{"descriptor":" Indoor   Court ","confidence":0.8},'
            '{"descriptor":"indoor court","confidence":0.7}]}'
        )
        with self.assertRaisesRegex(SchemaValidationError, "duplicate normalized"):
            DescriptorResponse.from_json(payload)

    def test_nonfinite_confidence_and_duplicate_json_key_are_rejected(self):
        with self.assertRaisesRegex(SchemaValidationError, "constant"):
            DescriptorResponse.from_json(
                '{"descriptors":[{"descriptor":"x","confidence":NaN}]}'
            )
        with self.assertRaisesRegex(SchemaValidationError, "duplicate JSON key"):
            DescriptorResponse.from_json(
                '{"descriptors":[],"descriptors":[]}'
            )

    def test_binary_label_rejects_bool_and_objects(self):
        for payload in ("true", "false", '{"label":1}', "1.0", "2"):
            with self.subTest(payload=payload):
                with self.assertRaises(SchemaValidationError):
                    BinaryLabel.from_json(payload)
        self.assertEqual(BinaryLabel.from_json("0").value, 0)
        self.assertEqual(BinaryLabel.from_json("1").value, 1)

    def test_qarm_judge_returns_single_binary_label(self):
        adapter = QueueAdapter("1")
        label = QARMJudge(adapter).label(
            descriptor="indoor court practice",
            title="Training ball",
            description="Low bounce",
            caption="Indoor court",
        )
        self.assertEqual(label, 1)
        self.assertIsNone(adapter.calls[0]["response_format"])
        self.assertIn("single JSON integer 0 or 1", str(adapter.calls[0]["messages"]))

    def test_captioner_sends_exactly_one_image(self):
        adapter = QueueAdapter("A textured ball on an indoor court.")
        caption = CMSACaptioner(adapter).caption("data:image/png;base64,AA==")
        self.assertEqual(caption, "A textured ball on an indoor court.")
        content = adapter.calls[0]["messages"][0]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 1)

    def test_injected_openai_compatible_client(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = OpenAICompatibleAdapter(
            api_key="test-placeholder",
            base_url="https://api.example.test/v1",
            deployment="external-deployment",
            client=client,
        )
        self.assertEqual(adapter.complete([{"role": "user", "content": "x"}]), "response")
        self.assertEqual(completions.kwargs["model"], "external-deployment")
        self.assertNotIn("api_key", completions.kwargs)
        self.assertNotIn("base_url", completions.kwargs)

    def test_default_openai_dependency_is_lazy(self):
        adapter = OpenAICompatibleAdapter(
            api_key="test-placeholder",
            base_url="https://api.example.test/v1",
            deployment="external-deployment",
        )
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("intentionally unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(RuntimeError, "optional 'api'"):
                adapter.complete([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
