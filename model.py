"""
Conditional LSTM-VAE for Mouse Trajectory Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ConditionalLSTMVAE(nn.Module):
    def __init__(self, config, num_users):
        """
        Conditional LSTM-VAE for generating mouse trajectories

        Args:
            config: Configuration object
            num_users: Number of unique users for embedding
        """
        super(ConditionalLSTMVAE, self).__init__()

        self.config = config
        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.latent_dim = config.LATENT_DIM
        self.num_layers = config.NUM_LAYERS

        # Condition embeddings
        self.user_embedding = nn.Embedding(num_users, config.USER_EMBED_DIM)
        self.time_embedding = nn.Embedding(4, config.TIME_PERIOD_EMBED_DIM)  # 4 time periods
        self.action_embedding = nn.Embedding(3, config.ACTION_TYPE_EMBED_DIM)  # 3 action types

        # Total condition dimension
        self.cond_dim = (config.USER_EMBED_DIM +
                        config.TIME_PERIOD_EMBED_DIM +
                        config.ACTION_TYPE_EMBED_DIM +
                        config.POSITION_DIM)

        # Encoder: LSTM + FC layers for mu and logvar
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_dim + self.cond_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )

        self.fc_mu = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)
        self.fc_logvar = nn.Linear(config.HIDDEN_DIM, config.LATENT_DIM)

        # Decoder: FC + LSTM + output layer
        self.fc_latent = nn.Linear(config.LATENT_DIM + self.cond_dim, config.HIDDEN_DIM)

        self.decoder_lstm = nn.LSTM(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )

        self.fc_output = nn.Linear(config.HIDDEN_DIM, self.input_dim)

    def encode_condition(self, condition):
        """
        Encode condition information into a single vector

        Args:
            condition: Dictionary with keys 'user_id', 'time_period', 'action_type',
                      'start_pos', 'end_pos'

        Returns:
            cond_vector: Tensor of shape (batch_size, cond_dim)
        """
        user_emb = self.user_embedding(condition['user_id'].squeeze(1))
        time_emb = self.time_embedding(condition['time_period'].squeeze(1))
        action_emb = self.action_embedding(condition['action_type'].squeeze(1))

        # Concatenate position information
        positions = torch.cat([condition['start_pos'], condition['end_pos']], dim=1)

        # Combine all condition embeddings
        cond_vector = torch.cat([user_emb, time_emb, action_emb, positions], dim=1)

        return cond_vector

    def encode(self, x, cond_vector):
        """
        Encode input sequence to latent distribution parameters

        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            cond_vector: Condition vector of shape (batch_size, cond_dim)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Expand condition to match sequence length
        cond_expanded = cond_vector.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate input with condition
        x_cond = torch.cat([x, cond_expanded], dim=2)

        # Pass through LSTM
        _, (h_n, _) = self.encoder_lstm(x_cond)

        # Use the last hidden state
        h = h_n[-1]  # (batch_size, hidden_dim)

        # Compute mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick

        Args:
            mu: Mean of shape (batch_size, latent_dim)
            logvar: Log variance of shape (batch_size, latent_dim)

        Returns:
            z: Sampled latent variable of shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, cond_vector, seq_len):
        """
        Decode latent variable to sequence

        Args:
            z: Latent variable of shape (batch_size, latent_dim)
            cond_vector: Condition vector of shape (batch_size, cond_dim)
            seq_len: Length of sequence to generate

        Returns:
            x_recon: Reconstructed sequence of shape (batch_size, seq_len, input_dim)
        """
        batch_size = z.shape[0]

        # Concatenate latent and condition
        z_cond = torch.cat([z, cond_vector], dim=1)

        # Project to hidden dimension
        h = self.fc_latent(z_cond)  # (batch_size, hidden_dim)

        # Expand to sequence length
        h_expanded = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)

        # Pass through decoder LSTM
        lstm_out, _ = self.decoder_lstm(h_expanded)

        # Generate output
        x_recon = self.fc_output(lstm_out)

        return x_recon

    def forward(self, x, condition):
        """
        Forward pass

        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            condition: Dictionary with condition information

        Returns:
            x_recon: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        batch_size, seq_len, _ = x.shape

        # Encode condition
        cond_vector = self.encode_condition(condition)

        # Encode
        mu, logvar = self.encode(x, cond_vector)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decode(z, cond_vector, seq_len)

        return x_recon, mu, logvar

    def generate(self, condition, seq_len, temperature=1.0):
        """
        Generate a trajectory from condition

        Args:
            condition: Dictionary with condition information
            seq_len: Length of sequence to generate
            temperature: Sampling temperature

        Returns:
            trajectory: Generated trajectory of shape (batch_size, seq_len, input_dim)
        """
        batch_size = condition['user_id'].shape[0]

        # Encode condition
        cond_vector = self.encode_condition(condition)

        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=cond_vector.device) * temperature

        # Decode
        trajectory = self.decode(z, cond_vector, seq_len)

        return trajectory


def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    """
    VAE loss function with per-feature weighting

    Args:
        x_recon: Reconstructed sequence
        x: Original sequence
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Per-feature weights to balance reconstruction loss
    # Feature order: [delta_t, delta_x, delta_y, speed, acceleration, button_encoded, state_encoded]
    feature_weights = torch.tensor([
        0.01,  # delta_t: reduce weight (variance too high)
        100.0, # delta_x: increase weight (critical for trajectory)
        100.0, # delta_y: increase weight (critical for trajectory)
        1.0,   # speed
        1.0,   # acceleration
        1.0,   # button
        1.0    # state
    ], device=x_recon.device)

    # Weighted MSE reconstruction loss
    squared_error = (x_recon - x) ** 2  # (batch, seq_len, features)
    weighted_error = squared_error * feature_weights.view(1, 1, -1)
    recon_loss = weighted_error.mean()

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.shape[0]  # Normalize by batch size

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test model
    config = Config()
    num_users = 50

    model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)

    # Create dummy data
    batch_size = 4
    seq_len = config.SEQUENCE_LENGTH

    x = torch.randn(batch_size, seq_len, config.INPUT_DIM).to(config.DEVICE)

    condition = {
        'user_id': torch.randint(0, num_users, (batch_size, 1)).to(config.DEVICE),
        'time_period': torch.randint(0, 4, (batch_size, 1)).to(config.DEVICE),
        'action_type': torch.randint(0, 3, (batch_size, 1)).to(config.DEVICE),
        'start_pos': torch.randn(batch_size, 2).to(config.DEVICE),
        'end_pos': torch.randn(batch_size, 2).to(config.DEVICE)
    }

    # Forward pass
    x_recon, mu, logvar = model(x, condition)

    print("Model Architecture:")
    print(model)
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")

    # Test generation
    generated = model.generate(condition, seq_len)
    print(f"\nGenerated trajectory shape: {generated.shape}")
