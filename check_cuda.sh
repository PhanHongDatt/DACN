source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
