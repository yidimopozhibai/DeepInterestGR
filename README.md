# DeepInterestGR

This repository provides the compact core implementation of **DeepInterestGR: Deep Interest Mining for Intent-Enriched Semantic IDs in Multimodal Generative Recommendation**. It implements the paper's three method components—Deep Contextual Interest Mining (DCIM), Cross-Modal Semantic Augmentation (CMSA), and the Quality-Aware Reinforcement Mechanism (QARM)—with inspectable interfaces, strict validation, and equation-level behavior.

## Concept-to-module map

| Paper concept | Module | Reference behavior |
| --- | --- | --- |
| CMSA caption path | `deepinterestgr.cmsa.CMSACaptioner` | Single-image, recommendation-oriented caption call through an injected OpenAI-compatible adapter |
| CMSA vision/fusion path | `deepinterestgr.cmsa.CMSAFusion` | Frozen `openai/clip-vit-large-patch14` vision path, direct trainable linear `W_v` and `W_f`, explicit dimensions, and matching batched text/image inputs |
| DCIM | `deepinterestgr.dcim.DCIMExtractor` | Ordered grounded item-side descriptors from title, description, and caption, with strict confidences and no fallback generation |
| Enriched text | `deepinterestgr.dcim.build_enriched_text` | Explicit `title → description → visual caption → ordered descriptors` construction before external Qwen3 embedding and CMSA fusion |
| QARM judge and aggregation | `deepinterestgr.qarm` | Binary descriptor labels and the confidence-weighted item score `q_i` |
| SID format and decoding | `deepinterestgr.sid` | Exactly three RQ codes plus an optional collision suffix, with catalog validation and decoding |
| Reward and GRPO equations | `deepinterestgr.qarm` | Exact/valid/LCP/quality reward breakdown, normalized advantages, and a pure clipped GRPO surrogate with KL penalty |

The repository is source-only: it contains the method-defining modules and offline equation checks, while datasets, pretrained weights, and large training artifacts remain external.

## Anonymous-review artifact scope

Large materials—including datasets, pretrained weights, full training pipelines, checkpoints, and logs—are not redistributed in this package. The compact review artifacts provide the method prompts, paper-reported seed/preprocessing disclosure and final settings, and reproduced anonymized case metadata in [`artifacts/`](artifacts/); the source modules provide the CMSA, DCIM, QARM, SID, and reward interfaces. External RQ-VAE and training systems remain explicit integration boundaries.

## Representation boundary

DeepInterestGR preserves the construction chain

```text
item evidence x_i → multimodal embedding e_i → SID_i
```

where the text branch is `title → description → CMSA visual caption → ordered DCIM descriptors`, and the continuous branch is frozen CLIP followed by `W_v`; `W_f` fuses both branches before the external RQ-VAE tokenizer. Evidence omitted before quantization is not explicitly encoded in SID semantics. DCIM descriptors are grounded item-side usage or purchase-context proxies—not verified psychological intent or personalized user state.

## Installation

Python 3.10 or newer is required. The core schemas, SID catalog, reward equations, and offline tests have no third-party runtime dependency.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For a source-tree check before installation, run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Optional integrations are installed separately:

```bash
python -m pip install -e '.[api]'  # default OpenAI-compatible SDK client
python -m pip install -e '.[ml]'   # PyTorch and Transformers CLIP loader
```

Imports remain usable without these optional packages. Tests inject fake API clients and in-memory tensor/module objects, so they make no network request and download no model or data.

## OpenAI-compatible API injection

Connection values are always external. Copy `.env.example` into your own environment-management workflow and pass the values explicitly:

```python
import os

from deepinterestgr import OpenAICompatibleAdapter

caption_adapter = OpenAICompatibleAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    deployment=os.environ["CMSA_CAPTION_DEPLOYMENT"],
)
dcim_adapter = OpenAICompatibleAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    deployment=os.environ["DCIM_DEPLOYMENT"],
)
qarm_adapter = OpenAICompatibleAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    deployment=os.environ["QARM_JUDGE_DEPLOYMENT"],
)
```

The paper uses Qwen2.5-VL-7B-Instruct for the CMSA caption path, Qwen2.5-7B-Instruct for DCIM and the QARM judge, and Qwen3-Embedding-4B at the external text-embedding boundary. The same small `CompletionAdapter` protocol is accepted by `CMSACaptioner`, `DCIMExtractor`, and `QARMJudge`, while keeping those model roles explicit.

## DCIM-to-CMSA construction

```python
from deepinterestgr import build_enriched_text

x_text = build_enriched_text(
    title=item_title,
    description=item_description,
    caption=visual_caption,
    descriptors=dcim_response,
)
# External boundary from the paper: e_text = Qwen3-Embedding-4B(x_text)
# Then: e_mm = cmsa_fusion(e_text, clip_image_inputs)
```

`build_enriched_text` preserves descriptor order and omits confidence values from the embedding input; DCIM confidences remain metadata consumed by QARM aggregation. The returned CMSA vector is the input boundary for the unchanged external RQ-VAE tokenizer.

## Minimal equation use

```python
import os

from deepinterestgr import (
    SIDCatalog,
    SemanticID,
    aggregate_quality,
    normalized_advantages,
    reward_breakdown,
)

lambda_valid = float(os.environ["LAMBDA_VALID"])
lambda_pref = float(os.environ["LAMBDA_PREF"])

catalog = SIDCatalog({
    "target": SemanticID(("a1", "b2", "c3")),
    "other": SemanticID(("a1", "b9", "c4")),
})
q_target = aggregate_quality([1, 0], [0.8, 0.2])
audit = reward_breakdown(
    candidate=("a1", "b2", "c3"),
    target_item_id="target",
    catalog=catalog,
    item_quality={"target": q_target},
    lambda_valid=lambda_valid,
    lambda_pref=lambda_pref,
)
advantages = normalized_advantages([audit.total, 0.0])
```

`lambda_valid`, `lambda_pref`, GRPO clipping epsilon, and KL beta are caller-required; the example reads the two format coefficients from the caller's environment rather than assigning unpublished values. QARM uses `alpha=0.5`, and group advantages use the paper-fixed `epsilon=1e-6`. Reward results expose every component for auditing.

## Validation boundaries

- DCIM and QARM parse exact JSON and reject extra fields, duplicate keys, non-finite values, boolean labels, and duplicate normalized descriptors.
- DCIM failures are surfaced rather than replaced with generic descriptors.
- Confidence aggregation accepts either all confidences or none; all missing means unit weights, while mixed missing/present or a zero denominator is rejected.
- Collision suffixes disambiguate catalog identity but never enter the three-code LCP term.
- Malformed generated SID sequences are treated as invalid without aborting reward computation; LCP inspects at most their first three RQ tokens.
- Descriptor quality is awarded only when the candidate decodes to the exact target item.
- CMSA keeps the vision encoder frozen and in evaluation mode even when the fusion module is switched to training mode.

## Anonymity note

To comply with ICLR double-blind review, this package does not link to access-tracked cloud storage or redistribute files whose metadata may reveal author identity. Large data and training artifacts can be released after the anonymous review period.
