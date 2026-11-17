import math
import unittest
import torch
import torch.nn as nn

# Import the functions from attention.py
from attention import precompute_rotary_emb, apply_rotary_emb, CausalSelfAttention


class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.dim = 32          # Dimension per head (must be even)
        self.max_positions = 64
        self.cache = precompute_rotary_emb(dim=self.dim, max_positions=self.max_positions)

    # -------------------------------------------------------------
    # 1. Tests for precompute_rotary_emb
    # -------------------------------------------------------------
    def test_precompute_shape_and_type(self):
        """Cache must have shape (max_positions, dim/2, 2)."""
        expected_shape = (self.max_positions, self.dim // 2, 2)
        self.assertEqual(self.cache.shape, expected_shape)
        self.assertFalse(self.cache.requires_grad)

    def test_precompute_position_zero(self):
        """At t=0, cos(0*theta) = 1 and sin(0*theta) = 0 for all frequencies."""
        cos_0 = self.cache[0, :, 0]
        sin_0 = self.cache[0, :, 1]
        self.assertTrue(torch.allclose(cos_0, torch.ones_like(cos_0)))
        self.assertTrue(torch.allclose(sin_0, torch.zeros_like(sin_0)))

    def test_trigonometric_identity(self):
        """cos^2(t*theta) + sin^2(t*theta) == 1 for every entry."""
        cos_vals = self.cache[..., 0]
        sin_vals = self.cache[..., 1]
        sum_of_squares = cos_vals ** 2 + sin_vals ** 2
        self.assertTrue(
            torch.allclose(sum_of_squares, torch.ones_like(sum_of_squares), atol=1e-5),
            "cos^2 + sin^2 did not equal 1!"
        )

    def test_first_frequency_value(self):
        """For i=0: theta_0 = 1 / 10000^0 = 1, so at t=1: cos(1) and sin(1)."""
        cos_t1_i0 = self.cache[1, 0, 0].item()
        sin_t1_i0 = self.cache[1, 0, 1].item()
        self.assertAlmostEqual(cos_t1_i0, math.cos(1.0), places=5)
        self.assertAlmostEqual(sin_t1_i0, math.sin(1.0), places=5)

    # -------------------------------------------------------------
    # 2. Tests for apply_rotary_emb
    # -------------------------------------------------------------
    def test_apply_identity_at_position_zero(self):
        """Tokens at t=0 should not change since rotation angle is 0."""
        # 1 token at position 0: shape (batch=2, seq=1, dim=32)
        x = torch.randn(2, 1, self.dim)
        rotated_x = apply_rotary_emb(x, self.cache)
        self.assertTrue(torch.allclose(x, rotated_x, atol=1e-5))

    def test_apply_norm_preservation(self):
        """Rotations are orthogonal; vector norm must be preserved."""
        x = torch.randn(4, 16, self.dim)
        rotated_x = apply_rotary_emb(x, self.cache)

        orig_norm = torch.linalg.norm(x, dim=-1)
        rot_norm = torch.linalg.norm(rotated_x, dim=-1)
        self.assertTrue(
            torch.allclose(orig_norm, rot_norm, atol=1e-5),
            "Vector norm was altered by rotary embedding!"
        )

    def test_apply_matches_manual_rotation(self):
        """Verify that apply_rotary_emb produces the exact 2D rotation:
        [x1*cos - x2*sin, x1*sin + x2*cos]."""
        t, i = 5, 0  # Position 5, pair 0
        x = torch.zeros(1, 6, self.dim)
        x[0, t, 0] = 1.0  # x1 = 1
        x[0, t, 1] = 2.0  # x2 = 2

        rotated_x = apply_rotary_emb(x, self.cache)

        cos_val = self.cache[t, i, 0]
        sin_val = self.cache[t, i, 1]
        expected_y1 = 1.0 * cos_val - 2.0 * sin_val
        expected_y2 = 1.0 * sin_val + 2.0 * cos_val

        self.assertAlmostEqual(rotated_x[0, t, 0].item(), expected_y1.item(), places=5)
        self.assertAlmostEqual(rotated_x[0, t, 1].item(), expected_y2.item(), places=5)

    def test_apply_relative_distance_invariance(self):
        """Dot product <R_m(q), R_n(k)> must equal <R_{m+s}(q), R_{n+s}(k)>."""
        q = torch.randn(1, 1, self.dim)
        k = torch.randn(1, 1, self.dim)

        # Place at m=2, n=5 (distance = 3)
        q_pos2 = apply_rotary_emb(q, self.cache[2:3])
        k_pos5 = apply_rotary_emb(k, self.cache[5:6])
        dot1 = (q_pos2 * k_pos5).sum()

        # Shift by s=10 to m=12, n=15 (distance still = 3)
        q_pos12 = apply_rotary_emb(q, self.cache[12:13])
        k_pos15 = apply_rotary_emb(k, self.cache[15:16])
        dot2 = (q_pos12 * k_pos15).sum()

        self.assertAlmostEqual(dot1.item(), dot2.item(), places=4)

    def test_apply_with_4d_tensor_multihead(self):
        """Must support 4D tensors (Batch, Num_Heads, Seq_Len, Head_Dim)."""
        B, nh, T, hs = 2, 4, 16, self.dim
        x_4d = torch.randn(B, nh, T, hs)
        
        # Cache has max_positions=64, input has T=16 (tests cache truncation)
        rotated_x = apply_rotary_emb(x_4d, self.cache)
        self.assertEqual(rotated_x.shape, (B, nh, T, hs))


if __name__ == '__main__':
    unittest.main()