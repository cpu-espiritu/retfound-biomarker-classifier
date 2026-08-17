#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
T = ['IRF', 'SRF', 'PED']


def layernorm(z, eps=1e-6):
    mu = z.mean(-1, keepdims=True)
    sd = z.std(-1, keepdims=True)
    return (z - mu) / (sd + eps)


def pool_fixed(X, how):
    """Parameter-free pooling: (n, tokens, d) -> (n, d)."""
    if how == 'mean':
        return X.mean(1)
    if how == 'max':
        return X.max(1)
    if how == 'topk':
        k = max(1, X.shape[1] // 20)                  # top 5% of tokens
        idx = np.argsort(X.sum(-1), axis=1)[:, -k:]
        return np.take_along_axis(X, idx[:, :, None], axis=1).mean(1)
    raise ValueError(how)


class AttnPool:
    """Gated attention pooling + linear head, numpy with analytic gradients.

        e_t = w . tanh(V z_t)      a = softmax(e)      p = sum_t a_t z_t
        logit = c . LN(p) + b
    """

    def __init__(self, d, L=64, lr=3e-3, epochs=120, wd=1e-4, seed=0):
        r = np.random.default_rng(seed)
        self.V = r.normal(0, 0.02, (d, L))
        self.w = r.normal(0, 0.02, L)
        self.c = np.zeros(d)
        self.b = 0.0
        self.lr, self.epochs, self.wd = lr, epochs, wd

    def _fwd(self, Z):
        Tn = np.tanh(Z @ self.V)                       # (n, t, L)
        e = Tn @ self.w                                # (n, t)
        e = e - e.max(1, keepdims=True)
        A = np.exp(e); A /= A.sum(1, keepdims=True)
        P = np.einsum('nt,ntd->nd', A, Z)              # (n, d)
        Pn = layernorm(P)
        return Tn, A, P, Pn, Pn @ self.c + self.b

    def fit(self, Z, y, batch=64):
        n = len(Z)
        st = {k: np.zeros_like(v) for k, v in
              dict(V=self.V, w=self.w, c=self.c, b=np.float64(0)).items()}
        vt = {k: np.zeros_like(v) for k, v in st.items()}
        rng = np.random.default_rng(0)
        t = 0
        for ep in range(self.epochs):
            for sl in np.array_split(rng.permutation(n), max(1, n // batch)):
                Zb, yb = Z[sl], y[sl]
                Tn, A, P, Pn, s = self._fwd(Zb)
                g = (1 / (1 + np.exp(-s)) - yb) / len(sl)        # (m,)
                dc = Pn.T @ g
                db = g.sum()
                dPn = g[:, None] * self.c[None, :]
                # LayerNorm backward
                mu = P.mean(-1, keepdims=True); sd = P.std(-1, keepdims=True) + 1e-6
                d = P.shape[-1]
                xh = (P - mu) / sd
                dP = (dPn - dPn.mean(-1, keepdims=True)
                      - xh * (dPn * xh).mean(-1, keepdims=True)) / sd
                dA = np.einsum('nd,ntd->nt', dP, Zb)
                dE = A * (dA - (A * dA).sum(1, keepdims=True))
                dTn = dE[:, :, None] * self.w[None, None, :]
                dU = dTn * (1 - Tn ** 2)
                dV = np.einsum('ntd,ntl->dl', Zb, dU)
                dw = np.einsum('ntl,nt->l', Tn, dE)
                gr = dict(V=dV + self.wd * self.V, w=dw,
                          c=dc + self.wd * self.c, b=db)
                t += 1
                for k, gk in gr.items():
                    st[k] = 0.9 * st[k] + 0.1 * gk
                    vt[k] = 0.999 * vt[k] + 0.001 * gk ** 2
                    mh = st[k] / (1 - 0.9 ** t); vh = vt[k] / (1 - 0.999 ** t)
                    setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + 1e-8))
        return self

    def decision(self, Z, batch=256):
        return np.concatenate([self._fwd(Z[i:i + batch])[4]
                               for i in range(0, len(Z), batch)])


def gradcheck(seed=0):
    r = np.random.default_rng(seed)
    Z = r.normal(size=(3, 5, 8)); y = np.array([1.0, 0.0, 1.0])
    m = AttnPool(8, L=4, seed=seed)
    Tn, A, P, Pn, s = m._fwd(Z)
    loss = lambda: float(np.mean(np.log1p(np.exp(-np.where(y == 1, 1, -1) * m._fwd(Z)[4]))))
    m2 = AttnPool(8, L=4, seed=seed); m2.epochs = 0
    worst = 0.0
    for name in ('V', 'w', 'c'):
        Pm = getattr(m, name); flat = Pm.reshape(-1)
        for j in r.choice(flat.size, min(6, flat.size), replace=False):
            o = flat[j]; h = 1e-6
            flat[j] = o + h; lp = loss()
            flat[j] = o - h; lm = loss()
            flat[j] = o
            worst = max(worst, abs((lp - lm) / (2 * h)))
    return worst


def main():
    ap = argparse.ArgumentParser(
        description='Does attention pooling beat mean pooling on frozen RETFound '
                    'tokens? Tests the dilution hypothesis: a small lesion averaged '
                    'with hundreds of normal tokens.')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--methods', nargs='*',
                    default=['mean', 'max', 'topk', 'attn'])
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--L', type=int, default=64)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    print(f'tokens {X.shape} {X.dtype}   {len(df)} scans')
    assert len(X) == len(df)

    pool = (df.split == 'pool').values
    test = (df.split == 'test').values
    rows = []

    for how in a.methods:
        for i, c in enumerate(T):
            y = df[f'label_{c}'].values
            oof = np.full(len(df), np.nan)
            for k in range(5):
                tr = pool & (df.fold != k).values
                va = pool & (df.fold == k).values
                if how == 'attn':
                    m = AttnPool(X.shape[2], L=a.L, epochs=a.epochs).fit(
                        np.asarray(X[tr], np.float32), y[tr].astype(float))
                    oof[va] = m.decision(np.asarray(X[va], np.float32))
                else:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), how))
                    Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), how))
                    sc = StandardScaler().fit(Ztr)
                    lr = LogisticRegression(C=0.01, max_iter=3000).fit(
                        sc.transform(Ztr), y[tr])
                    oof[va] = lr.predict_proba(sc.transform(Zva))[:, 1]
            m_ = pool
            rows.append(dict(pool=how, cls=c,
                             oof_auprc=AP(y[m_], oof[m_]),
                             oof_auroc=roc_auc_score(y[m_], oof[m_])))
            print(f'  {how:<6}{c:<5} OOF AUPRC {rows[-1]["oof_auprc"]:.3f}  '
                  f'AUROC {rows[-1]["oof_auroc"]:.3f}', flush=True)

    R = pd.DataFrame(rows)
    print('\n' + '=' * 52)
    print(R.pivot(index='pool', columns='cls', values='oof_auprc')
          .reindex(a.methods)[T].round(3).to_string())
    R.to_csv(ROOT / 'notebooks/pooling_comparison.csv', index=False)
    print('\n-> notebooks/pooling_comparison.csv')


if __name__ == '__main__':
    main()
