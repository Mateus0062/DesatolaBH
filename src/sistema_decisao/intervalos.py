# src/sistema_decisao/intervalos.py
# ITEM 8 (parte 1) — Quantificar incerteza.
# O recomendador atual dispara em desvios de 5-15%, mas o erro mediano do
# modelo é ~18%: os limiares estão dentro do ruído. Aqui produzimos uma BANDA
# de previsão calibrada (p10/p50/p90 ou conformal) para o recomendador só
# sinalizar quando o preço pedido sai da banda.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import TARGET, COLUNAS_EXCLUIR_MODELO
from config_extra import QUANTIS, NIVEL_CONFORMAL, HIPER_REGULARIZADO, \
    FEATURES_CATEGORICAS


def _Xy(df):
    y = np.log1p(df[TARGET])
    excluir = [c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
               if c in df.columns and c not in FEATURES_CATEGORICAS]
    X = df.drop(columns=excluir)
    for c in FEATURES_CATEGORICAS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, y


# ─── (A) regressão quantílica (p10 / p50 / p90) ─────────────────────────────
def treinar_quantis(df_tr, quantis=QUANTIS):
    from lightgbm import LGBMRegressor
    X_tr, y_tr = _Xy(df_tr)
    cats = [c for c in FEATURES_CATEGORICAS if c in X_tr.columns]
    hp = dict(HIPER_REGULARIZADO["lightgbm"])
    modelos = {}
    for q in quantis:
        m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=1200,
                          random_state=42, n_jobs=-1, verbose=-1, **hp)
        m.fit(X_tr, y_tr, categorical_feature=cats if cats else "auto")
        modelos[q] = m
    return modelos, list(X_tr.columns)


def prever_quantis(modelos, colunas, df_te):
    X_te, _ = _Xy(df_te)
    X_te = X_te[colunas]
    out = {f"p{int(q*100)}": np.expm1(m.predict(X_te))
           for q, m in modelos.items()}
    return pd.DataFrame(out, index=df_te.index)


# ─── (B) conformal split (garantia de cobertura) ───────────────────────────
class IntervaloConformal:
    """
    Conformal split no espaço log (erro multiplicativo).
    modelo_base: precisa ter .predict que devolve log-previsão.
    """
    def __init__(self, modelo_base, colunas, nivel=NIVEL_CONFORMAL):
        self.modelo = modelo_base
        self.colunas = colunas
        self.nivel = nivel
        self.qhat_ = None

    def calibrar(self, df_cal):
        X, y = _Xy(df_cal)
        X = X[self.colunas]
        log_pred = self.modelo.predict(X)
        scores = np.abs(y.values - log_pred)          # nonconformidade em log
        n = len(scores)
        nivel_ajustado = np.ceil((n + 1) * self.nivel) / n
        self.qhat_ = np.quantile(scores, min(nivel_ajustado, 1.0))
        print(f"  [item8] qhat conformal (log) = {self.qhat_:.4f} "
              f"→ banda ×{np.exp(self.qhat_):.2f} / ÷{np.exp(self.qhat_):.2f}")
        return self

    def prever(self, df_te):
        X, _ = _Xy(df_te)
        X = X[self.colunas]
        log_pred = self.modelo.predict(X)
        centro = np.expm1(log_pred)
        inf = np.expm1(log_pred - self.qhat_)
        sup = np.expm1(log_pred + self.qhat_)
        return pd.DataFrame({"inferior": inf, "previsto": centro,
                             "superior": sup}, index=df_te.index)


def treinar_base_log(df_tr):
    """Modelo pontual em log para o conformal (LightGBM regularizado)."""
    from lightgbm import LGBMRegressor
    X_tr, y_tr = _Xy(df_tr)
    cats = [c for c in FEATURES_CATEGORICAS if c in X_tr.columns]
    m = LGBMRegressor(n_estimators=1200, random_state=42, n_jobs=-1,
                      verbose=-1, **HIPER_REGULARIZADO["lightgbm"])
    m.fit(X_tr, y_tr, categorical_feature=cats if cats else "auto")
    return m, list(X_tr.columns)


def cobertura_empirica(banda, y_real):
    """Checa se ~NIVEL_CONFORMAL das observações caíram dentro da banda."""
    dentro = ((y_real >= banda["inferior"].values) &
              (y_real <= banda["superior"].values))
    return dentro.mean()


if __name__ == "__main__":
    from config import ITBI_FINAL
    df = pd.read_csv(ITBI_FINAL)
    # split temporal honesto: treino<=2021, calib=2022, teste>=2023
    tr = df[df["ano_transacao"] <= 2021]
    cal = df[df["ano_transacao"] == 2022]
    te = df[df["ano_transacao"] >= 2023]

    base, cols = treinar_base_log(tr)
    conf = IntervaloConformal(base, cols).calibrar(cal)
    banda = conf.prever(te)
    cob = cobertura_empirica(banda, te[TARGET].values)
    print(f"\nCobertura empírica no teste: {cob*100:.1f}% (alvo "
          f"{NIVEL_CONFORMAL*100:.0f}%)")