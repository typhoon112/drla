# DRLA Personal Experiment Environment

## Project-local conda environment

For the DRLA-MVP experiment, prefer the project-local conda environment:

```bash
conda env create -p /data1/luyifei/drla/.conda/drla-mvp -f /data1/luyifei/drla/configs/drla-mvp.environment.yml
conda run -p /data1/luyifei/drla/.conda/drla-mvp python -m pip install -r /data1/luyifei/drla/configs/drla-mvp.cuda-requirements.txt
conda run -p /data1/luyifei/drla/.conda/drla-mvp python -m pip install -r /data1/luyifei/drla/configs/drla-mvp.torch-requirements.txt
conda run -p /data1/luyifei/drla/.conda/drla-mvp python -m pip install -r /data1/luyifei/drla/configs/drla-mvp.requirements.txt
source /data1/luyifei/drla/scripts/activate_conda.sh
```

This keeps Python packages, pip cache, Hugging Face cache, and the environment itself under:

```text
/data1/luyifei/drla/.conda/drla-mvp
/data1/luyifei/drla/.cache
```

Recommended verification:

```bash
source /data1/luyifei/drla/scripts/activate_conda.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import transformers, datasets, accelerate; print(transformers.__version__, datasets.__version__, accelerate.__version__)"
```

## Project-local Python user base

This workspace must not mutate the shared Python environment. Use a project-local Python user base:

```bash
source /data1/luyifei/drla/scripts/env.sh
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user some_package==1.0.0
```

Packages installed with `--user` will be redirected by `PYTHONUSERBASE` to:

```text
/data1/luyifei/drla/.pyuser
```

Pip cache is redirected to:

```text
/data1/luyifei/drla/.cache/pip
```

Recommended check:

```bash
source /data1/luyifei/drla/scripts/env.sh
python3 -m site --user-base
python3 -m site --user-site
```

Rules:

- Do not run plain global `pip install`.
- Do not install into shared system Python.
- Prefer pinned package versions.
- Keep experiment outputs under `/data1/luyifei/drla/outputs`.
