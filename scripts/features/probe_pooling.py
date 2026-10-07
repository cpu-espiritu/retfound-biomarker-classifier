#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
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


CGRID = [0.001, 0.01, 0.1, 1.0]          # same grid the LP probes use


def fit_mean_baseline(Ztr, ytr, Zva, folds_tr, y_all, tr_mask):
    """Mean pooling + logistic head, C selected by inner CV on the training folds."""
    from sklearn.model_selection import StratifiedKFold
    best_C, best_s = CGRID[0], -np.inf
    if len(np.unique(ytr)) > 1 and np.bincount(ytr.astype(int)).min() >= 3:
        for C in CGRID:
            oof = np.full(len(ytr), np.nan)
            for itr, iva in StratifiedKFold(3, shuffle=True,
                                            random_state=0).split(Ztr, ytr):
                sc = StandardScaler().fit(Ztr[itr])
                m = LogisticRegression(C=C, max_iter=3000).fit(
                    sc.transform(Ztr[itr]), ytr[itr])
                oof[iva] = m.predict_proba(sc.transform(Ztr[iva]))[:, 1]
            s = AP(ytr, oof)
            if s > best_s:
                best_C, best_s = C, s
    sc = StandardScaler().fit(Ztr)
    m = LogisticRegression(C=best_C, max_iter=3000).fit(sc.transform(Ztr), ytr)
    return m.predict_proba(sc.transform(Zva))[:, 1], best_C


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
        n, t, d = Z.shape
        Zf = Z.reshape(-1, d)                          # BLAS-friendly 2-D views
        Tn = np.tanh(Zf @ self.V).reshape(n, t, -1)    # (n, t, L)
        e = Tn @ self.w                                # (n, t)
        e = e - e.max(1, keepdims=True)
        A = np.exp(e); A /= A.sum(1, keepdims=True)
        P = np.matmul(A[:, None, :], Z)[:, 0, :]       # (n, d)
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
                dA = np.matmul(Zb, dP[:, :, None])[:, :, 0]
                dE = A * (dA - (A * dA).sum(1, keepdims=True))
                dTn = dE[:, :, None] * self.w[None, None, :]
                dU = dTn * (1 - Tn ** 2)
                nb, tb, dfeat = Zb.shape        # not `db`: that holds the bias gradient
                dV = Zb.reshape(-1, dfeat).T @ dU.reshape(-1, dU.shape[-1])
                dw = Tn.reshape(-1, Tn.shape[-1]).T @ dE.reshape(-1)
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


def gradcheck(seed=0, n_probe=6):
    """Max |analytic - finite difference| over sampled parameters."""
    r = np.random.default_rng(seed)
    Z = r.normal(size=(4, 5, 8))
    y = np.array([1.0, 0.0, 1.0, 1.0])
    m = AttnPool(8, L=4, seed=seed)

    def loss():
        s_ = m._fwd(Z)[4]
        return float(np.mean(np.logaddexp(0, -np.where(y == 1, 1.0, -1.0) * s_)))

    # analytic gradient for the same mean-BCE objective, one full-batch step
    Tn, A, P, Pn, s_ = m._fwd(Z)
    g = (1 / (1 + np.exp(-s_)) - y) / len(Z)
    dc = Pn.T @ g
    dPn = g[:, None] * m.c[None, :]
    mu = P.mean(-1, keepdims=True); sd = P.std(-1, keepdims=True) + 1e-6
    xh = (P - mu) / sd
    dP = (dPn - dPn.mean(-1, keepdims=True)
          - xh * (dPn * xh).mean(-1, keepdims=True)) / sd
    dA = np.matmul(Z, dP[:, :, None])[:, :, 0]
    dE = A * (dA - (A * dA).sum(1, keepdims=True))
    dU = dE[:, :, None] * m.w[None, None, :] * (1 - Tn ** 2)
    ana = dict(V=Z.reshape(-1, 8).T @ dU.reshape(-1, 4),
               w=Tn.reshape(-1, 4).T @ dE.reshape(-1),
               c=dc,
               b=np.array([g.sum()]))

    worst = 0.0
    for name in ('V', 'w', 'c', 'b'):
        af = np.asarray(ana[name]).reshape(-1)
        if name == 'b':                       # a scalar attribute, not an array view
            o, h = m.b, 1e-6
            m.b = o + h; lp = loss()
            m.b = o - h; lm = loss()
            m.b = o
            worst = max(worst, abs((lp - lm) / (2 * h) - af[0]))
            continue
        flat = getattr(m, name).reshape(-1)
        for j in r.choice(flat.size, min(n_probe, flat.size), replace=False):
            o = flat[j]; h = 1e-6
            flat[j] = o + h; lp = loss()
            flat[j] = o - h; lm = loss()
            flat[j] = o
            worst = max(worst, abs((lp - lm) / (2 * h) - af[j]))
    return worst


def bias_sanity(seed=0):
    """The bias must respond to the data, not march by -lr every step.

    The gradient assembled in fit() was once shadowed by the feature dimension, so
    Adam saw a large constant and b landed on exactly -lr * steps whatever the
    labels were. Fitting the same features against a label and its complement must
    now move the bias in opposite directions.
    """
    r = np.random.default_rng(seed)
    Z = r.normal(size=(256, 8, 16))
    y = (r.random(256) < 0.3).astype(float)
    Z[:, 0, 0] += 0.9 * y
    lr, epochs = 1e-3, 20
    b_pos = AttnPool(16, L=8, lr=lr, epochs=epochs, seed=0).fit(Z, y).b
    b_neg = AttnPool(16, L=8, lr=lr, epochs=epochs, seed=0).fit(Z, 1 - y).b
    drift = -lr * epochs * max(1, len(Z) // 64)
    return dict(b_pos=float(b_pos), b_neg=float(b_neg),
                old_buggy_value=float(drift),
                data_dependent=bool(abs(b_pos - b_neg) > 1e-3))


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
    ap.add_argument('--out', default=str(ROOT / 'notebooks/pooling_comparison.csv'))
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    X = np.load(a.tokens, mmap_mode='r')
    print(f'tokens {X.shape} {X.dtype}   {len(df)} scans')
    assert len(X) == len(df)

    pool = (df.split == 'pool').values
    test = (df.split == 'test').values

    # resume: keep whatever a previous run already finished
    out = Path(a.out)
    rows = (pd.read_csv(out).to_dict('records') if out.exists() else [])
    done = {(r['pool'], r['cls']) for r in rows}
    if done:
        print(f'resuming, {len(done)} of {len(a.methods) * len(T)} already done')

    for how in a.methods:
        for i, c in enumerate(T):
            if (how, c) in done:
                continue
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
                    Ztr = layernorm(pool_fixed(np.asarray(X[tr], np.float32), how))
                    Zva = layernorm(pool_fixed(np.asarray(X[va], np.float32), how))
                    oof[va], chosen_C = fit_mean_baseline(Ztr, y[tr], Zva,
                                                          None, y, tr)
                    print(f'      {how} fold{k} C={chosen_C}', flush=True)
            m_ = pool
            rows.append(dict(pool=how, cls=c,
                             oof_auprc=AP(y[m_], oof[m_]),
                             oof_auroc=roc_auc_score(y[m_], oof[m_])))
            print(f'  {how:<6}{c:<5} OOF AUPRC {rows[-1]["oof_auprc"]:.3f}  '
                  f'AUROC {rows[-1]["oof_auroc"]:.3f}', flush=True)
            # flush after every cell so a kill never loses completed work
            np.save(out.with_name(f'oof_{how}_{c}.npy'), oof)
            pd.DataFrame(rows).to_csv(out, index=False)

    R = pd.DataFrame(rows)
    print('\n' + '=' * 52)
    piv = R.pivot(index='pool', columns='cls', values='oof_auprc')
    print(piv.reindex([m for m in a.methods if m in piv.index])[T].round(3).to_string())
    R.to_csv(out, index=False)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
