# Finetuning TabPFN 

## Install
After cloning the repo, do the following:

This code base requires at least Python 3.10. 

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install uv
# Local TabPFN install so we can change the model without reinstalling the package (e.g. for adapters)
cd custom_libs && git clone --branch finetune_tabpfn --single-branch https://github.com/LennartPurucker/TabPFN.git
# Install dependencies
uv pip install -e TabPFN && cd ..
uv pip install -r requirements.txt
```

### Do you want faster training on old GPUs?
Use the following to install flash attention 1 if you have a GPU like RTX2080 or T4.
Unsure if this still works, so be careful. 
```bash
uv pip uninstall torch torchvision torchaudio && uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

To test if flash attention is working, put the following code before you run TabPFN
```python
import torch.backends.cuda
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

## Examples
See `examples/toy_data_example_run.py` for an example of how to run the code.


## Developer Docs

* Add requirements to `requirements.txt`
* Change mypy and ruff settings in `pyproject.toml`
* Make sure the toy example runs without errors.  
