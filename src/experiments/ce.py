from ..methods.ce import train_ce
from ..utils.config import *

NOISE_TYPE = 'symmetric'
NOISE_RATES= [0.0, 0.20, 0.40, 0.60]
result_dicts_ce = []

def main():
	for noise_rate in NOISE_RATES:
		result_dict_ce = train_ce(
				batch_size=BATCH_SIZE,
				lr=LEARNING_RATE,
				num_epochs=NUM_EPOCHS,
				device=DEVICE,
				noise_type=NOISE_TYPE,
				noise_rate=noise_rate,
				warmup_epochs=WARMUP_EPOCHS
		)
		result_dicts_ce.append(result_dict_ce)

if __name__=='__main__':
		main()
