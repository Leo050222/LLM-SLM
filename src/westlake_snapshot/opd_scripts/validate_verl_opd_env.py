import random

import numpy as np
import torch


SEED = 20260806

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda:0")
a = torch.randn((512, 512), device=device, dtype=torch.float32)
b = torch.randn((512, 512), device=device, dtype=torch.float32)
c = a @ b
torch.cuda.synchronize()

checksum = c.double().sum().item()
if not np.isfinite(checksum):
    raise RuntimeError(f"non-finite CUDA checksum: {checksum}")

print(
    "WITNESS_OK",
    f"seed={SEED}",
    f"device={torch.cuda.get_device_name(0)}",
    f"torch={torch.__version__}",
    f"checksum={checksum:.10f}",
)
