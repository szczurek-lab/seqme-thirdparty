# seqme-thirdparty

Some models are not available through e.g., PyPI or Huggingface - only the git repository may be available. Other models licenses may not be compatible with seqme's license. Hence, why these models are not directly in the seqme package. Here we provide a repository for setting up such a third-party model.

## Ported models

This repository has several branches. Each branch is a model ported to be compatible with seqme's third-party interface.

## Getting started

An external model is compatible with seqme if it satisfies the following three requirements:

- Repository is accessible using `git clone`, e.g. a public repository or already on your local machine.
- Repository dependencies are installable using `pip install .`, e.g., by setup.py or pyproject.toml.
- Has a function with signature `Callable[[list[str], ...], np.ndarray]` where the first parameter is called `sequences`.

All branches in this repository satisfy these three requirements. To use the toy model in this branch, we define the function entry point, repository url and model directory.

```python
from seqme.models import ThirdPartyModel

model = ThirdPartyModel(
    entry_point="seqmetp.model:embed",
    repo_path="./plugins/thirdparty/main",
    repo_url="https://github.com/szczurek-lab/seqme-thirdparty",
    branch="main",
)

model(sequences=["SEQVENCE"])
```
