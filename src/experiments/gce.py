from ..methods.gce import train_gce
from ..utils.config import *

NOISE_TYPE = 'symmetric' # also test for asymmetric
NOISE_RATES = [0.20, 0.40]  # Test with 0, 20, 40% noise
GCE_Q = 0.7  # GCE robustness hyperparameter

result_dicts_gce = []

for noise_rate in NOISE_RATES:
    print(f"\nRunning GCE with noise_rate: {noise_rate}, q={GCE_Q}")
    result_dict_gce = train_gce(
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        noise_type=NOISE_TYPE,
        noise_rate=noise_rate,
        warmup_epochs=WARMUP_EPOCHS
    )
    result_dicts_gce.append(result_dict_gce)