from ..methods.elr import train_elr
from ..utils.config import *

# Noise settings for ELR experiments
NOISE_TYPE = 'symmetric'  # Can be 'symmetric' or 'asymmetric'
NOISE_RATES = [0.20, 0.40]  # Test on different levels of noise

# ELR-Specific Hyperparameters
LAMBDA_ELR = 3.0  # Regularization strength for ELR
BETA = 0.7        # EMA momentum for p_t updates


def main():
    # Store results
    result_dicts_elr = []

    # Run experiments for different noise rates
    for noise_rate in NOISE_RATES:
        print(f"\nRunning ELR with noise_rate: {noise_rate}")
        result_dict_elr = train_elr(
            batch_size=BATCH_SIZE,
            lr=BASE_LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            noise_type=NOISE_TYPE,
            noise_rate=noise_rate,
            warmup_epochs=WARMUP_EPOCHS,
            lambda_elr=LAMBDA_ELR,
            beta=BETA,
        )
        result_dicts_elr.append(result_dict_elr)

if __name__=='__main__':
    main()
