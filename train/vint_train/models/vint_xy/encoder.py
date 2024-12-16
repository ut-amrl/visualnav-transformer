import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEncoder(nn.Module):
    """
    Goal Encoder for ViNT GPS-adaptation architecture.
    Combines a fixed-size latent vector (3000 dimensions) with goal coordinates
    and processes it through a 2-layer MLP.
    """
    def __init__(self, input_dim, latent_dim=3000, mlp_hidden_dim=512, output_dim=256):
        """
        Args:
            input_dim (int): Dimension of the GPS coordinates (e.g., 2 for (x, y)).
            latent_dim (int): Fixed size of the learned latent vector.
            mlp_hidden_dim (int): Hidden size of the MLP layers.
            output_dim (int): Output size of the goal encoder (e.g., token embedding size).
        """
        super(GoalEncoder, self).__init__()
        
        # Learnable fixed-size tensor of size `latent_dim`
        self.latent_vector = nn.Parameter(torch.randn(latent_dim), requires_grad=True)
        
        self.out_features = output_dim
        # MLP for combining latent vector and GPS goal coordinates
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def extract_features(self, goal_xy):
        return self(goal_xy)

    def forward(self, x):
        """
        Forward pass for the goal encoder.

        Args:
            goal_coordinates (torch.Tensor): Tensor of shape (batch_size, input_dim)
                                             representing goal coordinates (e.g., (x, y)).

        Returns:
            torch.Tensor: Encoded goal of shape (batch_size, output_dim).
        """
        B = x.shape[0]

        # Expand the latent vector to match the batch size
        h = self.latent_vector.unsqueeze(0).expand(B, -1)
        
        # Concatenate latent vector with goal coordinates
        hx = torch.cat([h, x], dim=-1)
        
        y = self.mlp(hx)
        return y


# Example usage
if __name__ == "__main__":
    # Instantiate the goal encoder
    input_dim = 2  # For (x, y) goal coordinates
