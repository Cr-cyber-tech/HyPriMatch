from __future__ import annotations
import numpy as np
from .beaver import dot_beaver_shares

def _rand_nonzero(P: int, rng: np.random.Generator) -> int:
    # 生成 1..P-1
    r = int(rng.integers(1, P, dtype=np.int64))
    return r if r != 0 else 1

def eq_masked_zero_test_scalar_open_bit(
    aA: int, aB: int, bA: int, bB: int,
    tp,
    P: int,
    rng: np.random.Generator
) -> int:
    """
    Two-cloud masked zero test for equality:
      Δ = (a-b) mod P   (secret-shared)
      pick random r != 0 (secret-shared)
      z = Δ * r (secret-shared) via Beaver
      open z and return bit: 1 if z==0 else 0

    Leakage: only the equality bit (0/1) to BOTH clouds.
    """
    aA = int(aA) % P; aB = int(aB) % P
    bA = int(bA) % P; bB = int(bB) % P

    dA = (aA - bA) % P
    dB = (aB - bB) % P

    # sample nonzero r, then secret-share r as (rA,rB)
    r = _rand_nonzero(P, rng)
    rA = int(rng.integers(0, P, dtype=np.int64))
    rB = (r - rA) % P

    # reuse dot_beaver_shares for scalar mult (len=1 vectors)
    xA = np.array([dA], dtype=np.int64)
    xB = np.array([dB], dtype=np.int64)
    yA = np.array([rA], dtype=np.int64)
    yB = np.array([rB], dtype=np.int64)

    zA, zB = dot_beaver_shares(xA, xB, yA, yB, tp, P)  # scalar shares
    z = (int(zA) + int(zB)) % P

    return 1 if z == 0 else 0