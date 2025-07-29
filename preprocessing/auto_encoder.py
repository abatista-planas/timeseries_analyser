import concurrent.futures

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def fit_autoencoder(
    X_train_np,
    bottleneck_dim=8,
    device="cuda:0",
    epochs=30,
    batch_size=128,
    lr=1e-3,
    verbose=False,
):
    """
    Trains an autoencoder on X_train_np for the given bottleneck dimension.
    Returns the trained autoencoder and the final reconstruction loss (MSE).
    """
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True
    )
    model = Autoencoder(
        input_dim=X_train_np.shape[1], bottleneck_dim=bottleneck_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for (xb,) in loader:
            optimizer.zero_grad()
            x_recon = model(xb)
            loss = loss_fn(x_recon, xb)
            loss.backward()
            optimizer.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"[{device}] Epoch {epoch+1}/{epochs}  Loss: {loss.item():.5f}")
    model.eval()
    with torch.no_grad():
        recon = model(X_train_tensor)
        final_loss = loss_fn(recon, X_train_tensor).item()
    return model.cpu(), final_loss


def fit_autoencoder_with_validation(
    X_np,
    bottleneck_dim,
    device="cuda:0",
    epochs=30,
    batch_size=128,
    lr=1e-3,
    verbose=False,
    val_split=0.1,
    random_state=42,
):
    """
    Splits X_np into 90/10 (train/val), trains on train, computes and returns average loss on train and val.
    """
    X_train, X_val = train_test_split(
        X_np, test_size=val_split, random_state=random_state
    )
    model, _ = fit_autoencoder(
        X_train,
        bottleneck_dim=bottleneck_dim,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
    )
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        model = model.to(device)
        recon_train = model(X_train_tensor.to(device))
        recon_val = model(X_val_tensor.to(device))
        train_loss = nn.MSELoss()(
            recon_train, X_train_tensor.to(recon_train.device)
        ).item()
        val_loss = nn.MSELoss()(recon_val, X_val_tensor.to(recon_val.device)).item()
    avg_loss = 0.5 * (train_loss + val_loss)
    if verbose:
        print(
            f"Bottleneck {bottleneck_dim}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, Avg={avg_loss:.5f}"
        )
    return model.cpu(), avg_loss


def encode_with_autoencoder(model, X_np, device="cpu"):
    """
    Encodes X_np with a trained autoencoder, returning the bottleneck representation.
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        enc = model.encoder(X_tensor.to(next(model.parameters()).device))
        return enc.cpu().numpy()


def fit_autoencoder_for_search(args):
    """
    Worker function for parallel search (do not call directly).
    """
    X_np, bottleneck_dim, device, epochs, batch_size, lr, verbose = args
    print(f"Launching AE bottleneck={bottleneck_dim} on {device}")
    _, avg_loss = fit_autoencoder_with_validation(
        X_np, bottleneck_dim, device, epochs, batch_size, lr, verbose
    )
    return bottleneck_dim, avg_loss


def parallel_search_optimal_bottleneck(
    X_np,
    bottleneck_sizes=[2, 4, 8, 16, 32],
    epochs=30,
    batch_size=128,
    lr=1e-3,
    plot=True,
    verbose=False,
):
    """
    Runs autoencoder search for best bottleneck size, splitting jobs across all GPUs.
    Each candidate trains/validates on a 90/10 split and returns average loss.
    Returns: dict of bottleneck_size: avg_loss, and the best size.
    """
    n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, "No CUDA GPUs detected!"

    args_list = []
    for idx, bottleneck_dim in enumerate(bottleneck_sizes):
        device = f"cuda:{idx % n_gpus}"
        args_list.append(
            (X_np, bottleneck_dim, device, epochs, batch_size, lr, verbose)
        )

    losses_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        results = list(executor.map(fit_autoencoder_for_search, args_list))
        for bottleneck_dim, avg_loss in results:
            losses_dict[bottleneck_dim] = avg_loss

    # Plot if requested
    if plot:
        sizes = sorted(losses_dict.keys())
        losses = [losses_dict[s] for s in sizes]
        plt.figure(figsize=(6, 4))
        plt.plot(sizes, losses, marker="o")
        plt.xlabel("Bottleneck size (latent dim)")
        plt.ylabel("Avg. Reconstruction Loss (Train/Val)")
        plt.title("Autoencoder Bottleneck Size Search (Parallel GPUs, 90/10 split)")
        plt.grid(True)
        plt.show()

    best_size = min(losses_dict, key=losses_dict.get)
    print(
        f"\nBest bottleneck size: {best_size} (Avg Recon Loss={losses_dict[best_size]:.5f})"
    )
    return losses_dict, best_size
