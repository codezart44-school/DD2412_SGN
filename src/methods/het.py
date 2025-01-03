# def heteroscedastic_nll(logits, log_sigma2, labels):
#     """
#     Compute the Heteroscedastic NLL loss.
#     Args:
#         logits: Predicted mean logits (μ), shape [B, C].
#         log_sigma2: Predicted log-variance (log(σ²)), shape [B, C].
#         labels: Ground truth labels, shape [B].
#     Returns:
#         Loss value (scalar).
#     """
#     sigma2 = torch.exp(log_sigma2) + 1e-6 # Convert log(σ²) to σ² (ensure numerical stability)
#     one_hot_labels = F.one_hot(labels, num_classes=logits.size(1)).float()

#     # Compute NLL: log-variance penalty + scaled squared error
#     nll = 0.5 * ((logits - one_hot_labels) ** 2 / sigma2 + log_sigma2)
#     return nll.mean()

# def train_one_epoch_het(
#     model,
#     loader,
#     optimizer,
#     epoch
# ):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False):
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()

#         # Forward pass: get mean logits (μ) and log-variance (log(σ²))
#         mu, log_sigma2 = model(images)

#         # Compute HET loss
#         loss = heteroscedastic_nll(mu, log_sigma2, labels)
#         loss.backward()
#         optimizer.step()

#         # Compute accuracy
#         _, predicted = mu.max(1)  # Use μ for predictions
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         running_loss += loss.item()

#     accuracy = 100. * correct / total
#     return running_loss / len(loader), accuracy


# @torch.no_grad()
# def evaluate_het(model, loader):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in tqdm(loader, desc="Evaluating", leave=False):
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         # Forward pass: get mean logits (μ) and log-variance (log(σ²))
#         mu, log_sigma2 = model(images)

#         # Compute HET loss
#         loss = heteroscedastic_nll(mu, log_sigma2, labels)

#         # Compute accuracy
#         _, predicted = mu.max(1)  # Use μ for predictions
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         running_loss += loss.item()

#     accuracy = 100. * correct / total
#     return running_loss / len(loader), accuracy

# def train_het(
#     batch_size: int,
#     lr: float,
#     num_epochs: int,
#     device: torch.device,
#     noise_type: str = None,
#     noise_rate: float = 0.0,
#     warmup_epochs: int = 5
# ):
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Load data
#     train_loader, test_loader, input_shape = DataLoaderCIFAR10.get_loaders(
#         root=DATA_DIR,
#         download=False,
#         batch_size=batch_size,
#         num_workers=2,
#         noise_type=noise_type,
#         noise_rate=noise_rate
#     )

#     # Create HET model
#     model = WideResNetHET(depth=28, width_multiplier=2, num_classes=10, input_shape=input_shape).to(device)

#     # Optimizer
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

#     # Learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

#     # Training results
#     result_dict = {
#         'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [],
#         'hyperparams': {'batch_size': batch_size, 'lr': lr, 'num_epochs': num_epochs, 'noise_type': noise_type, 'noise_rate': noise_rate},
#     }

#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")

#         # Training step
#         train_loss, train_acc = train_one_epoch_het(model, train_loader, optimizer, epoch)

#         # Evaluation step
#         test_loss, test_acc = evaluate_het(model, test_loader)

#         # Save results
#         result_dict['train_loss'].append(train_loss)
#         result_dict['test_loss'].append(test_loss)
#         result_dict['train_acc'].append(train_acc)
#         result_dict['test_acc'].append(test_acc)

#         # Print progress
#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#         print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

#         # Update learning rate
#         scheduler.step()

#     print("Training complete!")
#     return result_dict