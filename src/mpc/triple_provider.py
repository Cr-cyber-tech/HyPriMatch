from pathlib import Path
import numpy as np
import json
from typing import Optional

class TripleProvider:
    """
    实验用：生成 Beaver 三元组 (a,b,c=a*b mod p)，并拆成两云 share。
    真实系统：这里应替换成预处理/OT/第三方产生的相关随机性。
    """
    def __init__(self, p: int, seed: int = 12345, cache_dir: Optional[Path] = None):
        self.p = int(p)
        self.rng = np.random.default_rng(seed)
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "meta.json").write_text(
                json.dumps({"p": self.p, "seed": seed}, indent=2),
                encoding="utf-8"
            )

    def _share_vec(self, x: np.ndarray):
        A = self.rng.integers(0, self.p, size=x.shape, dtype=np.int64)
        B = (x.astype(np.int64) - A) % self.p
        return A, B

    def gen_triple_vec(self, n: int):
        """
        生成长度 n 的向量三元组：a,b,c=a*b mod p，并返回两云 share。
        """
        a = self.rng.integers(0, self.p, size=(n,), dtype=np.int64)
        b = self.rng.integers(0, self.p, size=(n,), dtype=np.int64)

        # Python 大整数避免乘法溢出
        c = (a.astype(object) * b.astype(object)) % self.p
        c = np.array(c, dtype=np.int64)

        aA, aB = self._share_vec(a)
        bA, bB = self._share_vec(b)
        cA, cB = self._share_vec(c)
        return (aA, aB), (bA, bB), (cA, cB)
