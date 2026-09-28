import numpy as np

def dot_beaver_shares(xA, xB, yA, yB, triple_provider, p: int):
    """
    Beaver 安全点积（mod p）
    输出 zA,zB 使得 zA+zB = <x,y> mod p
    """
    n = xA.shape[0]
    (aA, aB), (bA, bB), (cA, cB) = triple_provider.gen_triple_vec(n)

    dA = (xA - aA) % p
    dB = (xB - aB) % p
    eA = (yA - bA) % p
    eB = (yB - bB) % p

    # 打开 d,e（两云交换后重构）
    d = (dA + dB) % p
    e = (eA + eB) % p

    d_obj = d.astype(object)
    e_obj = e.astype(object)

    # 云A加入 d*e 项，云B不加（保证 share 相加得到正确值）
    zA_terms = (cA.astype(object)
                + d_obj * bA.astype(object)
                + e_obj * aA.astype(object)
                + d_obj * e_obj) % p

    zB_terms = (cB.astype(object)
                + d_obj * bB.astype(object)
                + e_obj * aB.astype(object)) % p

    zA = int(np.array(zA_terms, dtype=object).sum() % p)
    zB = int(np.array(zB_terms, dtype=object).sum() % p)
    return zA, zB
