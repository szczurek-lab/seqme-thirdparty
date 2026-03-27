# seqme-thirdparty

Some models are not available through e.g., PyPI or Huggingface - only the git repository may be available. Other models licenses may not be compatible with seqme's license. Hence, why these models are not directly in the seqme package. Here we provide a repository for setting up such a third-party model.

## Ported models

| Model | Repository | Description |
|-------|------------|-------------|
| AMPlify | [seqme-amplify](https://github.com/szczurek-lab/seqme-amplify) | Attentive deep learning model for antimicrobial peptide (AMP) prediction |
| amPEPpy | [seqme-amPEPpy](https://github.com/szczurek-lab/seqme-amPEPpy) | Random forest classifier for antimicrobial peptide prediction using global protein sequence descriptors |
| ESM-IF1 | [seqme-esmif1](https://github.com/szczurek-lab/seqme-esmif1) | Inverse folding model that generates amino acid sequences from fixed 3D protein backbones |

## Getting started

An external model is compatible with seqme if it is setup using [uv](https://docs.astral.sh/uv/) (lockfile, python version defined), and defines an entry point (function).
https://github.com/szczurek-lab/seqme-esmif1
Setup a project using:

```bash
uv init --package hello-model
```

Run the model:

```python
import seqme as sm

model = sm.models.ThirdPartyModel(
    entry_point="hello_model.model:embed",
    path="../thirdparty/hello-model",
    url="https://github.com/szczurek-lab/seqme-thirdparty",
    branch="main",
)

model(sequences=["SEQVENCE"])
```
