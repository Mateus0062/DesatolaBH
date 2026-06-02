import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, OUTPUTS_MODELS_TRAIN2, TARGET,
                    COLUNAS_EXCLUIR_MODELO, HIPERPARAMETROS,
                    QUANTIS, NIVEL_CONFORMAL, classificar_regime)


def _Xy(df):
    y = df[TARGET].copy()
    aux = ['_data', '_periodo']
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] + aux
                         if c in df.columns])
    return X, y


class RecomendadorBanda:
    """Recomendador com faixa de preço justo (quantis + conformal)."""

    def __init__(self, quantis=QUANTIS, nivel_conformal=NIVEL_CONFORMAL):
        self.quantis = sorted(quantis)
        self.nivel = nivel_conformal
        self.modelos_q = {}
        self.colunas = None
        self.qhat_ = None  # margem conformal (espaço log)

    # ─── treino ─────────────────────────────────────────────────────────────
    def treinar(self, df_treino, df_calib):
        """df_treino: ajusta os quantis. df_calib: calibra a banda conformal.
        Ambos em log(R$)."""
        from lightgbm import LGBMRegressor
        X_tr, y_tr = _Xy(df_treino)
        self.colunas = list(X_tr.columns)
        y_tr_log = np.log1p(y_tr)

        hp = dict(HIPERPARAMETROS['lightgbm'])
        hp.pop('n_estimators', None)
        for q in self.quantis:
            m = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1500,
                              random_state=42, n_jobs=-1, verbose=-1, **hp)
            m.fit(X_tr, y_tr_log)
            self.modelos_q[q] = m
        print(f"  {len(self.quantis)} modelos de quantil treinados "
              f"({self.quantis})")

        # calibração conformal: usa o quantil mediano como ponto e mede o erro
        # absoluto em log no conjunto de calibração.
        self._calibrar(df_calib)
        return self

    def _calibrar(self, df_calib):
        X_cal, y_cal = _Xy(df_calib)
        X_cal = X_cal[self.colunas]
        q_med = self._mediana()
        pred_log = self.modelos_q[q_med].predict(X_cal)
        scores = np.abs(np.log1p(y_cal.values) - pred_log)
        n = len(scores)
        nivel_aj = np.ceil((n + 1) * self.nivel) / n
        self.qhat_ = float(np.quantile(scores, min(nivel_aj, 1.0)))
        print(f"  margem conformal (log) qhat = {self.qhat_:.4f} "
              f"-> fator ×÷{np.exp(self.qhat_):.2f}")

    def _mediana(self):
        # quantil mais próximo de 0.5 entre os treinados
        return min(self.quantis, key=lambda q: abs(q - 0.5))

    # ─── previsão da faixa ───────────────────────────────────────────────────
    def prever_faixa(self, df_imovel):
        """Retorna DataFrame com inferior/previsto/superior (em R$).
        A faixa é o MAIOR intervalo entre (p10,p90) e a banda conformal —
        assim cobrimos tanto a dispersão dos quantis quanto a calibração."""
        if isinstance(df_imovel, dict):
            df_imovel = pd.DataFrame([df_imovel])
        X = _Xy(df_imovel)[0] if TARGET in df_imovel.columns else \
            df_imovel.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO
                                    if c in df_imovel.columns], errors='ignore')
        X = X[self.colunas]

        q_lo, q_med, q_hi = self.quantis[0], self._mediana(), self.quantis[-1]
        log_med = self.modelos_q[q_med].predict(X)
        prev = np.expm1(log_med)

        # banda dos quantis
        inf_q = np.expm1(self.modelos_q[q_lo].predict(X))
        sup_q = np.expm1(self.modelos_q[q_hi].predict(X))
        # banda conformal (em torno da mediana)
        inf_c = np.expm1(log_med - self.qhat_)
        sup_c = np.expm1(log_med + self.qhat_)

        inferior = np.minimum(inf_q, inf_c)
        superior = np.maximum(sup_q, sup_c)
        return pd.DataFrame({'inferior': inferior, 'previsto': prev,
                             'superior': superior}, index=df_imovel.index)

    # ─── recomendação ────────────────────────────────────────────────────────
    def analisar(self, df_imovel, preco_pedido):
        faixa = self.prever_faixa(df_imovel).iloc[0]
        inf, prev, sup = faixa['inferior'], faixa['previsto'], faixa['superior']
        largura_rel = (sup - inf) / prev * 100

        if preco_pedido > sup:
            acao, prio = 'REDUZIR PREÇO', 'ALTA'
            just = (f"Preço pedido acima do teto da faixa estimada "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). Mesmo considerando a incerteza "
                    f"do modelo, está caro.")
            sugerido = sup
        elif preco_pedido < inf:
            acao, prio = 'POSSÍVEL OPORTUNIDADE / SUBDECLARAÇÃO', 'MÉDIA'
            just = (f"Preço pedido abaixo do piso da faixa estimada "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). Pode ser barganha real ou "
                    f"valor subdeclarado — investigar.")
            sugerido = inf
        else:
            acao, prio = 'DENTRO DA FAIXA DE MERCADO', 'BAIXA'
            just = (f"Preço dentro do intervalo estimado "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). A diferença para o valor "
                    f"central é indistinguível da incerteza do modelo — sem ação.")
            sugerido = preco_pedido

        return {
            'preco_pedido': preco_pedido, 'previsto': prev,
            'faixa_inferior': inf, 'faixa_superior': sup,
            'largura_rel_%': largura_rel, 'acao': acao, 'prioridade': prio,
            'preco_sugerido': sugerido, 'justificativa': just,
        }

    def relatorio(self, a):
        print("\n" + "=" * 70)
        print("ANÁLISE DO IMÓVEL (com banda de incerteza)")
        print("=" * 70)
        print(f"  Preço pedido : R$ {a['preco_pedido']:>13,.2f}")
        print(f"  Estimado     : R$ {a['previsto']:>13,.2f}")
        print(f"  Faixa justa  : R$ {a['faixa_inferior']:>13,.2f} – "
              f"R$ {a['faixa_superior']:,.2f}  (±{a['largura_rel_%']/2:.0f}%)")
        print(f"  Recomendação : {a['acao']}  [{a['prioridade']}]")
        print(f"  Sugerido     : R$ {a['preco_sugerido']:,.2f}")
        print(f"  {a['justificativa']}")

    # ─── persistência ────────────────────────────────────────────────────────
    def salvar(self, caminho):
        with open(caminho, 'wb') as f:
            pickle.dump({'quantis': self.quantis, 'nivel': self.nivel,
                         'modelos_q': self.modelos_q, 'colunas': self.colunas,
                         'qhat_': self.qhat_}, f)
        print(f"  Recomendador salvo: {caminho}")

    @classmethod
    def carregar(cls, caminho):
        with open(caminho, 'rb') as f:
            d = pickle.load(f)
        r = cls(quantis=d['quantis'], nivel_conformal=d['nivel'])
        r.modelos_q = d['modelos_q']; r.colunas = d['colunas']; r.qhat_ = d['qhat_']
        return r


def cobertura_empirica(rec, df_teste):
    """Valida: ~NIVEL_CONFORMAL das transações devem cair dentro da faixa."""
    faixa = rec.prever_faixa(df_teste)
    y = df_teste[TARGET].values
    dentro = (y >= faixa['inferior'].values) & (y <= faixa['superior'].values)
    return dentro.mean()


if __name__ == '__main__':
    df = pd.read_csv(ITBI_FINAL)
    df['_data'] = pd.to_datetime(df['data_transacao'])
    regime = classificar_regime(df['_data'])

    # treino<=2023 (regime novo incluso) ; calibração e teste em 2024.
    df_2024 = df[df['ano_transacao'] == 2024].sort_values('_data')
    corte = len(df_2024) // 2
    df_calib = df_2024.iloc[:corte]      # 1ª metade de 2024 -> calibra
    df_teste = df_2024.iloc[corte:]      # 2ª metade -> valida cobertura
    df_treino = df[df['ano_transacao'] <= 2023]

    print("Treinando recomendador com banda (LightGBM quantílico)...")
    rec = RecomendadorBanda().treinar(df_treino, df_calib)

    cob = cobertura_empirica(rec, df_teste)
    print(f"\nCobertura empírica no teste: {cob*100:.1f}% "
          f"(alvo {NIVEL_CONFORMAL*100:.0f}%)")

    rec.salvar(OUTPUTS_MODELS_TRAIN2 / 'recomendador_banda.pkl')

    # demonstração: 3 cenários de preço pedido para um imóvel
    exemplo = df_teste.iloc[[0]]
    prev = rec.prever_faixa(exemplo).iloc[0]['previsto']
    print("\nDEMONSTRAÇÃO (imóvel de exemplo):")
    for fator in (1.30, 1.00, 0.70):
        rec.relatorio(rec.analisar(exemplo, prev * fator))
