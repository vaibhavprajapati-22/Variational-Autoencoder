from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import torch
import math
import torchvision


def load_dataset(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    data_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


class Encoder(nn.Module):
    def __init__(self, fan_in, fan_out, hidden_dim):
        super(Encoder, self).__init__()

        self.fan_in = fan_in
        self.hidden_dim = hidden_dim
        self.fan_out = fan_out

        self.hidden_layer = nn.Linear(fan_in, hidden_dim)

        self.mu_layer = nn.Linear(hidden_dim, fan_out)
        self.log_std_layer = nn.Linear(hidden_dim, fan_out)

        self.unmasked_layer = nn.Linear(hidden_dim, fan_out * fan_out)
        self.mask = torch.tril(torch.ones(fan_out, fan_out), diagonal=-1)
        self.register_buffer('mask_const', self.mask)

    def forward(self, x):
        hidden_output = F.relu(self.hidden_layer(x))
        mu_output = self.mu_layer(hidden_output)
        log_std_output = self.log_std_layer(hidden_output)

        unmasked_layer_output = self.unmasked_layer(hidden_output)
        unmasked_layer_output = unmasked_layer_output.view(-1, self.fan_out, self.fan_out)

        stanndard_deviation_output = torch.exp(log_std_output)
        lower_triangular_matrix = unmasked_layer_output * self.mask_const + torch.diag_embed(stanndard_deviation_output)

        eps = torch.randn_like(stanndard_deviation_output)
        z = mu_output + torch.bmm(lower_triangular_matrix, eps.view(-1, self.fan_out, 1)).view(-1, self.fan_out)
        return z, eps, log_std_output


class Decoder(nn.Module):
    def __init__(self, fan_in, fan_out, hidden_dim):
        super(Decoder, self).__init__()

        self.fan_in = fan_in
        self.hidden_dim = hidden_dim
        self.fan_out = fan_out

        self.hidden_layer = nn.Linear(fan_in, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, fan_out)

    def forward(self, z):
        hidden_output = F.relu(self.hidden_layer(z))
        output = F.sigmoid(self.output_layer(hidden_output))

        return output


class VariationalAutoEncoder(nn.Module):
    def __init__(self, fan_in=28 * 28, hidden_dim=1024, latent_dim=2):
        super(VariationalAutoEncoder, self).__init__()

        self.fan_in = fan_in
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(fan_in, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, fan_in, hidden_dim)

    def forward(self, x):
        z, eps, log_std_output = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, eps, log_std_output


def compute_negative_evidence_lower_bound(x, x_reconstructed, z, eps, log_std):
    log_2pi = math.log(2 * math.pi)

    # Reconstruction Loss
    reconstruction_loss = torch.nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction="sum"
    )

    # Variational Posterior Term
    latent_loss_q = 0.5 * torch.sum(eps ** 2 + log_std + log_2pi)

    # Prior Term
    prior_loss = 0.5 * torch.sum(z ** 2 + log_2pi)

    # ELBO and Negative ELBO
    elbo = -reconstruction_loss - prior_loss + latent_loss_q
    negative_elbo = -elbo

    # Average Negative ELBO
    batch_size = x.size(0)
    average_negative_elbo = negative_elbo / batch_size

    return average_negative_elbo


def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train():
    VAE = VariationalAutoEncoder()

    data_dir = "data"
    checkpoint_dir = "checkpoints"
    images_generated_dir = "images_generated"
    batch_size = 128
    epochs = 50
    warmup_steps = 5
    max_steps = 50
    max_lr = 1e-3
    min_lr = 0.01 * max_lr
    lr = max_lr
    num_samples = 16

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(images_generated_dir, exist_ok=True)
    data_loader = load_dataset(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAE.to(device)

    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)

    VAE.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(data_loader):
            lr = get_lr(epoch, warmup_steps, max_lr, min_lr, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad()
            x, labels = batch
            x = x.to(device)
            x = x.view(-1, 28 * 28)
            x_reconstructed, z, eps, log_std_output = VAE(x)
            loss = compute_negative_evidence_lower_bound(x, x_reconstructed, z, eps, log_std_output)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(data_loader)
        print(f"{epoch + 1} / {epochs} ---> Loss : {epoch_loss:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': VAE.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

        with torch.no_grad():
            sample = torch.randn(num_samples, VAE.latent_dim).to(device)
            sample = VAE.decoder(sample).cpu()
            torchvision.utils.save_image(sample.view(num_samples, 1, 28, 28),
                                         os.path.join(images_generated_dir, f"images_generated_{epoch + 1}.png"))


if __name__ == "__main__":
    train()