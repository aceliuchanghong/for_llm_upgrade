import torch
import os
import sys

from transformers import PreTrainedModel

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from pretrain.minimind.model.LMConfig import LMConfig


def detect_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No GPU available. Please check your CUDA installation.")


if __name__ == "__main__":
    """
    uv run pretrain/minimind/model/model.py
    """
    detect_gpu()
    xx = LMConfig()
    print(f"{xx}")
