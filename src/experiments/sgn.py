from ..methods.sgn import train_sgn
from ..utils.config import *

NOISE_TYPE = 'asymmetric'
NOISE_RATES= [0.4]

ALPHA = 0.995
EMA_DECAY = 0.99

# Ablation Study
disable_lr = False # NOTE Disable one at a time!
disable_lc = False

def main():
    result_dicts_sgn = []
    for noise_rate in NOISE_RATES:
        result_dict_sgn = train_sgn(
            batch_size=BATCH_SIZE,
            lr=BASE_LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            noise_type=NOISE_TYPE,
            noise_rate=noise_rate,
            alpha=ALPHA,
            ema_decay=EMA_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            disable_lr=disable_lr,
            disable_lc=disable_lc,
        )
    result_dicts_sgn.append(result_dict_sgn)

if __name__=='__main__':
    main()

