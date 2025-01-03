# NOTE Was not included in final report!

# # Configuration for HET
# BATCH_SIZE = 128
# NUM_EPOCHS = 100
# BASE_LEARNING_RATE = 0.1
# WARMUP_EPOCHS = 5
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NOISE_TYPE = 'symmetric' # Also test for asymmetric
# NOISE_RATES = [0.00]  # Test with 0, 20, 40% Noise rate

# # Results storage
# result_dicts_het = []

# # Running HET
# for noise_rate in NOISE_RATES:
#     print(f"\nRunning HET with noise_rate: {noise_rate}")
#     result_dict_het = train_het(
#         batch_size=BATCH_SIZE,
#         lr=BASE_LEARNING_RATE,
#         num_epochs=NUM_EPOCHS,
#         device=DEVICE,
#         noise_type=NOISE_TYPE,
#         noise_rate=noise_rate,
#         warmup_epochs=WARMUP_EPOCHS
#     )
#     result_dicts_het.append(result_dict_het)

# # Print completion message
# print("HET training and evaluation completed!")