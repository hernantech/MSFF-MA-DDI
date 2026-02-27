"""
clifford_algebra.py
-------------------
Clifford algebra Cl(3,0) implementation for asymmetric DDI prediction.

Basis (8 elements):
    index  0  :  1        (scalar)
    index  1  :  e1       (grade-1)
    index  2  :  e2       (grade-1)
    index  3  :  e3       (grade-1)
    index  4  :  e12      (grade-2)
    index  5  :  e13      (grade-2)
    index  6  :  e23      (grade-2)
    index  7  :  e123     (grade-3, pseudoscalar)

Signature (3,0):  e_i * e_i = +1  for i = 1,2,3
Anticommutativity: e_i * e_j = -e_j * e_i  for i ≠ j

Key property for asymmetric DDI:
    geometric_product(a, b) ≠ geometric_product(b, a)   in general
    ⟹  score(A→B) ≠ score(B→A)  is structurally guaranteed
"""

import torch
import torch.nn as nn


# ── Structure constants ────────────────────────────────────────────────────────

def build_structure_constants() -> torch.Tensor:
    """Compute the (8, 8, 8) structure constant tensor for Cl(3,0).

    S[i, j, k] = coefficient of basis[k] in the product basis[i] * basis[j].

    Algorithm:
    1. Represent each basis element as a sorted tuple of generator indices {0,1,2}.
    2. Compute the product by concatenating tuples.
    3. Bubble-sort to canonical order, tracking sign from transpositions.
    4. Cancel adjacent equal generators using e_k^2 = +1.
    5. Look up the resulting tuple in the basis.
    """
    # Basis elements as sorted tuples of generator indices (0-indexed generators)
    basis = [
        (),          # 0: 1
        (0,),        # 1: e1
        (1,),        # 2: e2
        (2,),        # 3: e3
        (0, 1),      # 4: e12
        (0, 2),      # 5: e13
        (1, 2),      # 6: e23
        (0, 1, 2),   # 7: e123
    ]
    basis_to_idx = {b: i for i, b in enumerate(basis)}

    S = torch.zeros(8, 8, 8, dtype=torch.float32)

    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            combined = list(a) + list(b)
            sign = 1

            # Bubble sort → canonical (sorted) order, track sign flips
            n = len(combined)
            for pass_num in range(n):
                for pos in range(n - 1 - pass_num):
                    if combined[pos] > combined[pos + 1]:
                        combined[pos], combined[pos + 1] = (
                            combined[pos + 1], combined[pos]
                        )
                        sign *= -1

            # Cancel adjacent equal generators: e_k^2 = +1 in Cl(3,0)
            reduced = []
            idx = 0
            while idx < len(combined):
                if (idx + 1 < len(combined)
                        and combined[idx] == combined[idx + 1]):
                    # e_k * e_k = +1  →  sign unchanged
                    idx += 2
                else:
                    reduced.append(combined[idx])
                    idx += 1

            result_basis = tuple(reduced)
            k = basis_to_idx[result_basis]
            S[i, j, k] = sign

    return S


def get_product_matrix(device: str = 'cpu') -> torch.Tensor:
    """Return the (64, 8) matrix for efficient batched geometric product.

    The geometric product c = a ⊙ b can be computed as:
        outer = a[..., :, None] * b[..., None, :]   → (..., 8, 8)
        flat  = outer.reshape(..., 64)
        c     = flat @ M                             → (..., 8)

    where M = S.reshape(64, 8), i.e. M[i*8+j, k] = S[i,j,k].
    This is a SINGLE batched matrix multiplication — fast on GPU.
    """
    S = build_structure_constants()
    return S.reshape(64, 8).to(device)


# ── Grade projection helpers ───────────────────────────────────────────────────

GRADE_INDICES = {
    0: [0],         # scalar
    1: [1, 2, 3],   # vectors
    2: [4, 5, 6],   # bivectors
    3: [7],         # pseudoscalar
}


def grade_projection(mv: torch.Tensor, grade: int) -> torch.Tensor:
    """Project a multivector onto its grade-k component.

    mv: (..., 8)
    returns: (..., 8) with only the grade-k components non-zero
    """
    mask = torch.zeros(8, dtype=mv.dtype, device=mv.device)
    for idx in GRADE_INDICES[grade]:
        mask[idx] = 1.0
    return mv * mask


def scalar_part(mv: torch.Tensor) -> torch.Tensor:
    """Extract the scalar (grade-0) component: (..., 8) → (...)"""
    return mv[..., 0]


def reverse(mv: torch.Tensor) -> torch.Tensor:
    """Clifford conjugate / reverse: flip sign of grades 2 and 3.

    Rev(e_{i1...ik}) = e_{ik...i1} = (-1)^{k(k-1)/2} e_{i1...ik}
        grade 0: sign = +1
        grade 1: sign = +1
        grade 2: sign = -1  (k=2: (-1)^1 = -1)
        grade 3: sign = -1  (k=3: (-1)^3 = -1)
    """
    sign = torch.ones(8, dtype=mv.dtype, device=mv.device)
    for idx in GRADE_INDICES[2] + GRADE_INDICES[3]:
        sign[idx] = -1.0
    return mv * sign


# ── Core module ────────────────────────────────────────────────────────────────

class CliffordProduct(nn.Module):
    """Stateless module that computes batched Cl(3,0) geometric products.

    Registers the (64, 8) product matrix as a buffer so it moves to GPU
    automatically with .to(device).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('M', get_product_matrix())  # (64, 8)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute a ⊙ b for a, b ∈ Cl(3,0).

        a, b : (..., 8)
        returns : (..., 8)
        """
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)    # (..., 8, 8)
        flat  = outer.reshape(*a.shape[:-1], 64)      # (..., 64)
        return flat @ self.M                           # (..., 8)

    def triple(self, a: torch.Tensor, t: torch.Tensor,
               b: torch.Tensor) -> torch.Tensor:
        """Compute (a ⊙ t) ⊙ b  — the sandwich product for scoring.

        a, t, b : (..., 8)
        returns : (..., 8)
        """
        return self.forward(self.forward(a, t), b)

    def commutator(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Commutator: [a, b] = a⊙b - b⊙a.  Non-zero iff a and b don't commute."""
        return self.forward(a, b) - self.forward(b, a)


# ── Verification ───────────────────────────────────────────────────────────────

def verify_algebra():
    """Sanity-check the geometric product against known Cl(3,0) identities."""
    prod = CliffordProduct()
    e = torch.eye(8)  # basis elements as one-hot vectors

    def gp(i, j):
        return prod(e[i].unsqueeze(0), e[j].unsqueeze(0)).squeeze(0)

    print('Cl(3,0) verification:')

    # e1*e1 = 1
    r = gp(1, 1)
    assert abs(r[0].item() - 1.0) < 1e-6 and r[1:].abs().max() < 1e-6, f'e1*e1 failed: {r}'
    print('  e1*e1 = 1  ✓')

    # e1*e2 = e12
    r = gp(1, 2)
    assert abs(r[4].item() - 1.0) < 1e-6, f'e1*e2 failed: {r}'
    print('  e1*e2 = e12  ✓')

    # e2*e1 = -e12  (non-commutativity)
    r = gp(2, 1)
    assert abs(r[4].item() + 1.0) < 1e-6, f'e2*e1 failed: {r}'
    print('  e2*e1 = -e12  ✓  (asymmetry confirmed)')

    # e12*e12 = -1
    r = gp(4, 4)
    assert abs(r[0].item() + 1.0) < 1e-6, f'e12*e12 failed: {r}'
    print('  e12*e12 = -1  ✓')

    # e1*e2*e3 = e123
    r = prod(gp(1, 2).unsqueeze(0), e[3].unsqueeze(0)).squeeze(0)
    assert abs(r[7].item() - 1.0) < 1e-6, f'e1*e2*e3 failed: {r}'
    print('  e1*e2*e3 = e123  ✓')

    # e1*e2 ≠ e2*e1  (core asymmetry guarantee)
    ab = gp(1, 2)
    ba = gp(2, 1)
    assert not torch.allclose(ab, ba), 'e1*e2 == e2*e1 — symmetry leak!'
    print('  e1*e2 ≠ e2*e1  ✓  (non-commutativity holds)')

    print('All checks passed.')


if __name__ == '__main__':
    verify_algebra()
