import os

# Prevent OpenMP runtime collision between PyTorch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

try:
    import faiss  # noqa: F401
except ImportError:
    pass
