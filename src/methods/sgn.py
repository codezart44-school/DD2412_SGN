import os
import torch
import torch.optim as optim
import torch.nn.functional as F  # provides functions for activation, loss etc.
from tqdm import tqdm
import math

from ..models.wide_resnet_28_2 import create_model
from ..utils.config import *
from ..datasets.dataloader.cifar import DataLoaderCIFAR10



def label_smoothing(y, num_classes, smoothing=0.001):
    # y is a tensor of shape [B] with class indices
    # Convert to one-hot
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    with torch.no_grad():
        y_smoothed = (1.0 - smoothing) * y_onehot + smoothing * torch.ones_like(y_onehot) / num_classes
    return y_smoothed

def helmert_matrix(n):
    # Helmert matrix construction for ilr transform
    # Equivalent to the code shown in the TF version, but now in torch
    # Creates a (n x (n-1)) matrix
    H = torch.zeros((n, n), dtype=torch.float)
    for i in range(n):
        for j in range(n):
            if i > 0:
                if j < i:
                    H[i,j] = 1.0 / torch.sqrt(torch.tensor(i*(i+1.), dtype=torch.float))
                elif j == i:
                    H[i,j] = -i / torch.sqrt(torch.tensor(i*(i+1.), dtype=torch.float))
    # Remove first row
    return H[1:,:]

def ilr_inv(p):
    # p in the simplex, shape [B, K], returns ilr_inv vector shape [B, K-1]
    # clr_inv(p) = log(p) - mean(log(p))
    # ilr_inv uses the helmert matrix
    log_p = torch.log(p)
    clr = log_p - log_p.mean(dim=1, keepdim=True)
    # Apply inverse of ilr: clr * V (where V = H^T)
    K = p.shape[1]
    H = helmert_matrix(K).to(p.device)
    # clr is [B, K], H^T is [K, K-1], so output is [B, K-1]
    return clr @ H.T

def ilr_forward(z):
    # Inverse of ilr_inv. Given z in R^{K-1}, we find p in simplex.
    # p = softmax(H z) basically:
    # We need the inverse transform of ilr: p = clr_forward(z V^T)
    K_1 = z.shape[1]
    K = K_1 + 1
    H = helmert_matrix(K).to(z.device)
    # z is [B, K-1], convert back to clr space:
    # clr = z @ H^T^-1 = z @ H^+ (since H transforms clr->ilr)
    # But we defined ilr_inv as clr @ H.T. Thus, ilr is essentially clr*H
    # So to invert: clr = z @ H (since ilr_inv used @H.T)
    # Actually we defined ilr_inv as clr @ H.T. So ilr: z = clr @ H
    # => clr = z @ H^-1. But since H is orthonormal (in ideal math), H^-1 = H.T
    # We'll just do symmetrical approach similar to TF code:
    # Let's store H once.
    # Actually, let's store and recall from doc: p = softmax( z * H', axis=1 )
    # Wait, from sgn code: ilr_inv does z @ H^T. So inverse:
    # ilr_forward: we have z, we want p: we do p = softmax(z @ H, dim=1)? Actually we must do the inverse of what we did.
    # If ilr_inv: p -> clr = log p - mean(log p); clr @ H^T = z
    # => clr = z @ H. Now given z, clr = z @ H. Then p = softmax(clr)
    clr = z @ H
    p = torch.softmax(clr, dim=1)
    return p


def update_ema(model, ema_model, decay=0.9999):
    with torch.no_grad():
        msd = model.state_dict()
        emsd = ema_model.state_dict()
        for k in msd.keys():
            emsd[k].copy_(emsd[k] * decay + (1.0 - decay) * msd[k])


def multivariate_normal_nll(x, mean, r=None):
    # x, mean: [B, K-1]
    # If r is None, treat it as identity matrix (no loss reweighting)
    diff = x - mean
    if r is None:
        mahalanobis = (diff * diff).sum(dim=1)
        logdet = 0  # log|I| = 0
    else:
        rr = (r * r).sum(dim=1, keepdim=True)  # r^T r
        scalar = (r * diff).sum(dim=1, keepdim=True)  # r^T diff
        denom = 1.0 + rr
        inv_diff = diff - (r * scalar) / denom
        mahalanobis = (inv_diff * diff).sum(dim=1)
        logdet = torch.log(denom.squeeze(1))

    k = diff.shape[1]
    c = k * math.log(2 * math.pi)

    log_prob = -0.5 * (c + logdet + mahalanobis)
    return -log_prob.mean()  # negative log-likelihood


def train_one_epoch_sgn(
      model,
      ema_model,
      loader,
      optimizer,
      epoch,
      num_epochs,
      alpha=0.995,
      ema_decay=0.9999,
      disable_lr=False,
      disable_lc=False,
    ):

    model.train()
    ema_model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_steps = len(loader)

    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Label smoothing
        num_classes = model.fc_mu.out_features + 1
        smoothed = label_smoothing(labels, num_classes)

        # Transform to ilr space
        t = ilr_inv(smoothed)

        # Forward pass
        mu, r = model(images)
        with torch.no_grad():
            mu_ema, r_ema = ema_model(images)

        # Compute shift factor
        current_step = (epoch * num_steps) + step
        total_steps = num_epochs * num_steps
        exponent = float(current_step)/float(total_steps)

        #---------------------DISABLE LC-------------------------#

        # print('LC DEACTIVATION:', disable_lc)
        if disable_lc == True:
          factor = 0  # NOTE set factor to 0 to disable LC (Label Correction)
        else:
          factor = 1.0 - (alpha ** exponent)

        #---------------------DISABLE LC-------------------------#


        # Compute shifted mean
        mean = mu + factor * (t - mu_ema)


        # Compute NLL
        #---------------------DISABLE LR-------------------------#

        # print('Lr DEACTIVATION:', disable_lr)
        if disable_lr == True:
            loss = multivariate_normal_nll(t, mean, None)  # NOTE set r = None to disable LR (Loss Reweighting)
        else:
            loss = multivariate_normal_nll(t, mean, r)

        #---------------------DISABLE LR-------------------------#

        loss.backward()
        optimizer.step()

        # Update EMA model
        update_ema(model, ema_model, decay=ema_decay)

        # For accuracy, convert mu to probabilities and predict
        probs = ilr_forward(mu)
        _, predicted = probs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

    accuracy = 100. * correct / total
    return running_loss / num_steps, accuracy


@torch.no_grad()
def evaluate_sgn(model, ema_model, loader):
    model.eval()
    ema_model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        num_classes = model.fc_mu.out_features + 1
        smoothed = label_smoothing(labels, num_classes)
        t = ilr_inv(smoothed)

        # We use the EMA model for evaluation as it tends to generalize better
        mu, r = ema_model(images)
        # No label correction at test time, factor = 1 (fully corrected)
        # Actually, at test time we want no shift: i.e., we just trust the EMA predictions as true mean
        # So we don't add any (t - mu_ema) shift.
        # mean = mu (since factor=0 means no correction)
        # For simplicity, set factor=0 at test: mean = mu
        mean = mu

        loss = multivariate_normal_nll(t, mean, r)

        # Compute accuracy
        probs = ilr_forward(mu)
        _, predicted = probs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

def train_sgn(
      batch_size: int,
      lr: float,
      num_epochs: int,
      device: torch.device,
      noise_type: str = None,
      noise_rate: float = 0.0,
      alpha=0.995,
      ema_decay=0.9999,
      warmup_epochs=5,
      disable_lr=False,
      disable_lc=False,
    ):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, test_loader, input_shape = DataLoaderCIFAR10.get_loaders(
        root=DATA_DIR,
        download=False,
        batch_size=batch_size,
        num_workers=2,
        noise_type=noise_type,
        noise_rate=noise_rate
    )

    # Create SGN model
    model = create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=input_shape, version=1, sgn=True).to(device)

    # Create EMA model
    ema_model = create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=input_shape, version=1, sgn=True).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False


    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # -------------------- CHANGES START HERE --------------------
    # Define warmup parameters
    base_lr = lr  # This is the lr you passed in, for example 0.1
    start_lr = base_lr * 0.1  # Starting at a lower LR for warmup (e.g. 0.01 if base_lr=0.1)

    # Create a cosine annealing scheduler for after warmup
    # Note: T_max is the number of epochs to run the cosine schedule
    # after warmup. So if total epochs = num_epochs, warmup is 5,
    # then T_max = num_epochs - warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    # Define a helper function for warmup
    def warmup_lr_scheduler(optimizer, epoch, warmup_epochs, start_lr, base_lr):
        # Linearly scale LR from start_lr to base_lr over warmup_epochs
        lr = start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # -------------------- CHANGES END HERE --------------------

    result_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'hyperparams': {
            'batch_size': batch_size,
            'lr': lr,
            'num_epochs': num_epochs,
            'noise_type': noise_type,
            'noise_rate': noise_rate,
        },
    }

    for epoch in range(num_epochs):
        # -------------------- CHANGES START HERE --------------------
        # Adjust the learning rate based on whether we're in warmup or not
        if epoch < warmup_epochs:
            # During warmup: manually set LR
            warmup_lr_scheduler(optimizer, epoch+1, warmup_epochs, start_lr, base_lr)
        else:
            # After warmup: use cosine annealing
            scheduler.step()
        # -------------------- CHANGES END HERE --------------------

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch_sgn(
            model,
            ema_model,
            train_loader,
            optimizer,
            epoch, num_epochs,
            alpha=alpha,
            ema_decay=ema_decay,
            disable_lr=disable_lr,
            disable_lc=disable_lc,
          )
        test_loss, test_acc = evaluate_sgn(model, ema_model, test_loader)

        result_dict['train_loss'].append(train_loss)
        result_dict['test_loss'].append(test_loss)
        result_dict['train_acc'].append(train_acc)
        result_dict['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, checkpoint_path)

    print("Training complete!")
    return result_dict


