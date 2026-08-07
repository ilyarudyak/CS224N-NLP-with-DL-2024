import torch
import matplotlib.pyplot as plt


import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    """
    A simple implementation of a Vanilla RNN in PyTorch.
    We do NOT use nn.RNN or any other built-in RNN modules. 
    Instead, we implement the forward pass manually.
    """


    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recurrent_matrix: torch.Tensor,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Input-to-hidden transformation.
        self.input_layer = nn.Linear(
            input_size,
            hidden_size,
            bias=True,
        )

        # Hidden-to-hidden matrix initialized with our chosen matrix.
        # Again, we do NOT use nn.RNN or nn.RNNCell
        # self.recurrent_matrix = W_hh
        self.recurrent_matrix = nn.Parameter(
            recurrent_matrix.clone()
        )

    def forward(self, inputs, initial_hidden=None):
        """
        inputs shape: [sequence_length, input_size]
        """
        sequence_length = inputs.size(0)

        if initial_hidden is None:
            hidden = torch.zeros(
                self.hidden_size,
                device=inputs.device,
            )
        else:
            hidden = initial_hidden

        # Retain the gradient of the initial hidden state.
        hidden.retain_grad()

        hidden_states = [hidden]

        for timestep in range(sequence_length):
            input_contribution = self.input_layer(inputs[timestep])

            # W_hh @ h_{t-1}; using matrix-vector multiplication.
            recurrent_contribution = (
                self.recurrent_matrix @ hidden
            )

            hidden = torch.tanh(
                input_contribution + recurrent_contribution
            )

            # Hidden states are non-leaf tensors, so retain their gradients.
            hidden.retain_grad()
            hidden_states.append(hidden)

        return hidden_states
    

class GradientExperiments:
    """
    Class to demonstrate Vanishing and Exploding Gradients in PyTorch.
    """
    def __init__(self, 
                 input_size=0, 
                 hidden_size=8, 
                 output_size=0):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Set the device to MPS if available, otherwise fallback to CPU
        self.device = self._get_device()

        # Create a vanishing matrix with a spectral radius less than 1
        self.vanishing_matrix = self.create_init_matrix(
            spectral_radius=0.5,  # Less than 1 for vanishing gradients
            eigenvalue_spread=0.0,  # All eigenvalues equal to spectral_radius
            seed=42,
        ).to(self.device)

    def _get_device(self) -> torch.device:
        """
        Get mps device if available, otherwise fallback to CPU.
        """
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def create_init_matrix(self, 
                           spectral_radius: float = 1.0,
                           eigenvalue_spread: float = 0.0,
                           seed: int = 42) -> torch.Tensor:
        """
        Create a real symmetric matrix with a specified spectral radius.

        spectral_radius:
            Largest absolute eigenvalue.

        eigenvalue_spread:
            Optional spread around the spectral radius.
            0.0 gives all eigenvalues with magnitude spectral_radius.
        """

        # Set the random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Generate a random orthogonal basis Q
        random_matrix = torch.randn(
            self.hidden_size,
            self.hidden_size,
            generator=generator,
        )
        Q, _ = torch.linalg.qr(random_matrix)

        # Create a tensor with all eigenvalues equal to the spectral radius 
        if eigenvalue_spread == 0.0:
            # torch.full fills a tensor with the specified value
            eigenvalues = torch.full(
                (self.hidden_size,),
                spectral_radius,
            ) # [hidden_size]
        # Create a tensor with eigenvalues spread around the spectral radius
        else:
            eigenvalues = torch.linspace(
                spectral_radius - eigenvalue_spread,
                spectral_radius + eigenvalue_spread,
                self.hidden_size,
            )

        # Scale the eigenvalues to ensure the maximum absolute eigenvalue equals the spectral radius
        maximum_eigenvalue = eigenvalues.abs().max()
        eigenvalues = eigenvalues * spectral_radius / maximum_eigenvalue

        # Create a diagonal matrix D with the eigenvalues on the diagonal
        D = torch.diag(eigenvalues)

        # We use a spectral theorem: W = Q @ D @ Q.T, where Q is orthogonal.
        W = (
            Q
            @ D
            @ Q.T
        )

        return W

    def inspect_matrix_powers(
        self,
        recurrent_matrix: torch.Tensor,
        number_of_steps: int = 20,
    ) -> list[float]:
        """
        Return the Frobenius norm of W^k for k = 1, ..., number_of_steps.
        The Frobenius norm is a matrix norm defined as the square root of 
        the sum of the absolute squares of its elements. 
        """
        matrix_power = torch.eye(
            recurrent_matrix.size(0),
            dtype=recurrent_matrix.dtype,
            device=recurrent_matrix.device,
        )

        power_norms = []

        for _ in range(1, number_of_steps + 1):
            matrix_power = matrix_power @ recurrent_matrix
            power_norms.append(matrix_power.norm().item())

        # Plot the norms to visualize the vanishing or exploding effect
        plt.plot(range(1, number_of_steps + 1), power_norms)
        plt.xlabel("Step")
        plt.ylabel("Frobenius Norm")
        plt.title("Matrix Powers Norms")
        plt.show()

        return power_norms