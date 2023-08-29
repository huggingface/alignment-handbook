# The RLHF Handbook

Robust recipes for RLHF

## Developing
To view this locally, run the following (need to point to a specific language if previewing only one section, preview/`build_doc` does not use the `--language` flag):
```shell
doc-builder preview rlhf-handbook {docs_dir} --not_python_module
```
Example `docs_dir` is `~/Documents/HuggingFace/dev/rlhf-handbook/chapters/en`

## Installation
Create a new conda environment with:
```shell
conda create -n rlhf-handbook python=3.10
```
Install the limited requirements with
```shell
pip install -r requirements.txt
```