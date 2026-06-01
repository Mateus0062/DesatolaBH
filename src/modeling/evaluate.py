import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_TABLES,
                    TARGET, COLUNAS_EXCLUIR_MODELO, classificar_regime)


# ─── métricas (item 7: não só MAPE, que estoura no barato) ──────────────────
def calcular_metricas(y_true, y_pred, nome_modelo=None):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs((y_true - y_pred) / y_true) * 100
    rmsle = np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None))
                             - np.log1p(y_true)) ** 2))
    m = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': ape.mean(),
        'MdAPE': np.median(ape),
        'RMSLE': rmsle,
        'R²': r2_score(y_true, y_pred),
    }
    if nome_modelo is not None:
        m = {'Modelo': nome_modelo, **m}
    return m


def metricas_por_faixa(y_true, y_pred,
                       faixas=((0, 1e5), (1e5, 3e5), (3e5, 6e5),
                               (6e5, 1e6), (1e6, np.inf))):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs((y_true - y_pred) / y_true) * 100
    linhas = []
    for lo, hi in faixas:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        rot = f">{lo/1e3:.0f}k" if hi == np.inf else f"{lo/1e3:.0f}-{hi/1e3:.0f}k"
        linhas.append({'faixa': rot, 'n': int(m.sum()),
                       'MAPE': ape[m].mean(), 'MdAPE': np.median(ape[m])})
    return pd.DataFrame(linhas)


def _Xy(df):
    y = df[TARGET].copy()
    # '_data'/'_periodo' são auxiliares do backtest; nunca podem virar feature.
    auxiliares = ['_data', '_periodo']
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] + auxiliares
                         if c in df.columns])
    return X, y


# ─── baseline (item 7: sempre reporte um piso simples) ──────────────────────
class BaselinePrecoM2:
    """preço = (R$/m² mediano do bairro+tipo no treino) * área."""
    def fit(self, df_tr):
        pm2 = df_tr[TARGET] / df_tr['area_construida_m2']
        chave = df_tr['bairro'].astype(str) + '|' + df_tr['tipo_construtivo'].astype(str)
        self.tab_ = pm2.groupby(chave).median()
        self.global_ = pm2.median()
        return self

    def predict(self, df_te):
        chave = df_te['bairro'].astype(str) + '|' + df_te['tipo_construtivo'].astype(str)
        pm2 = chave.map(self.tab_).fillna(self.global_)
        return pm2.values * df_te['area_construida_m2'].values


# ─── funções treina-e-prevê plugáveis (cada uma é um "experimento") ─────────
def fn_baseline(df_tr, df_te):
    return BaselinePrecoM2().fit(df_tr).predict(df_te)


def fn_lightgbm(df_tr, df_te):
    """LightGBM regularizado, alvo em log(R$). Usa HIPERPARAMETROS do config."""
    from lightgbm import LGBMRegressor
    from config import HIPERPARAMETROS
    X_tr, y_tr = _Xy(df_tr)
    X_te, _ = _Xy(df_te)
    X_te = X_te[X_tr.columns]
    hp = dict(HIPERPARAMETROS['lightgbm'])
    mdl = LGBMRegressor(**hp, random_state=42, n_jobs=-1, verbose=-1)
    mdl.fit(X_tr, np.log1p(y_tr))
    return np.expm1(mdl.predict(X_te))


def fn_xgboost(df_tr, df_te):
    from xgboost import XGBRegressor
    from config import HIPERPARAMETROS
    X_tr, y_tr = _Xy(df_tr)
    X_te, _ = _Xy(df_te)
    X_te = X_te[X_tr.columns]
    mdl = XGBRegressor(**HIPERPARAMETROS['xgboost'], random_state=42,
                       n_jobs=-1, verbosity=0)
    mdl.fit(X_tr, np.log1p(y_tr))
    return np.expm1(mdl.predict(X_te))

def fn_ensemble(df_tr, df_te):
    return (fn_lightgbm(df_tr, df_te) + fn_xgboost(df_tr, df_te)) / 2.0


def fn_separado_por_tipo(df_tr, df_te, base=fn_lightgbm):
    """ITEM 6: um modelo para AP, outro para CA."""
    pred = np.zeros(len(df_te))
    for tipo in df_te['tipo_construtivo'].unique():
        m_tr = df_tr['tipo_construtivo'] == tipo
        m_te = (df_te['tipo_construtivo'] == tipo).values
        treino = df_tr[m_tr] if m_tr.sum() >= 500 else df_tr
        pred[m_te] = base(treino, df_te[m_te])
    return pred


# ─── backtest rolling-origin (item 2/7) ─────────────────────────────────────
def backtest_rolling(df, treinar_e_prever, anos_corte, nome='modelo',
                     apenas_regime_novo=False, silencioso=False):
    if not silencioso:
        print("\n" + "=" * 80)
        print(f"BACKTEST {'(REGIME NOVO) ' if apenas_regime_novo else ''}— {nome}")
        print("=" * 80)
    df = df.copy()
    df['_data'] = pd.to_datetime(df['data_transacao'])

    if apenas_regime_novo:
        df = df[classificar_regime(df['_data']) == 'novo'].copy()
        df['_periodo'] = df['_data'].dt.to_period('Q')
        periodos = sorted(df['_periodo'].unique())
        pares = [(periodos[i - 1], periodos[i]) for i in range(1, len(periodos))]
        col = '_periodo'
    else:
        pares = [(Y, Y + 1) for Y in anos_corte]
        col = 'ano_transacao'

    res = []
    for ate, alvo in pares:
        tr = df[df[col] <= ate]
        te = df[df[col] == alvo]
        if len(te) < 100 or len(tr) < 500:
            continue
        y_pred = treinar_e_prever(tr, te)
        m = calcular_metricas(te[TARGET].values, y_pred)
        m['treino_ate'] = str(ate); m['testa'] = str(alvo); m['n'] = len(te)
        res.append(m)
        if not silencioso:
            print(f"  treino<= {ate} | teste {alvo}: MAPE {m['MAPE']:5.1f}% | "
                  f"MdAPE {m['MdAPE']:5.1f}% | RMSLE {m['RMSLE']:.3f} | "
                  f"R² {m['R²']:.3f} (n={m['n']:,})")

    res = pd.DataFrame(res)
    if len(res) and not silencioso:
        print(f"\n  RESUMO ({nome}) — média ± desvio:")
        for c in ['MAPE', 'MdAPE', 'RMSLE', 'R²']:
            print(f"    {c:6s}: {res[c].mean():7.3f} ± {res[c].std():.3f}")
    return res


def tabela_comparativa(df, modelos, anos, apenas_regime_novo=False):
    """Roda o MESMO backtest para vários modelos e imprime uma tabela única
    com média ± desvio. modelos = dict {nome: funcao_treina_preve}."""
    titulo = "REGIME NOVO" if apenas_regime_novo else "SÉRIE INTEIRA"
    print("\n" + "#" * 80)
    print(f"#  COMPARATIVO DE MODELOS — {titulo}")
    print("#" * 80)
    linhas = []
    for nome, fn in modelos.items():
        res = backtest_rolling(df, fn, anos, nome,
                               apenas_regime_novo=apenas_regime_novo,
                               silencioso=True)
        if len(res):
            linhas.append({
                'Modelo': nome,
                'MAPE': f"{res['MAPE'].mean():.2f} ± {res['MAPE'].std():.2f}",
                'MdAPE': f"{res['MdAPE'].mean():.2f} ± {res['MdAPE'].std():.2f}",
                'RMSLE': f"{res['RMSLE'].mean():.3f}",
                'R²': f"{res['R²'].mean():.3f}",
                '_ord': res['MdAPE'].mean(),
            })
    tab = pd.DataFrame(linhas).sort_values('_ord').drop(columns='_ord')
    print("\n" + tab.to_string(index=False))
    melhor = tab.iloc[0]['Modelo']
    print(f"\n  → Melhor por MdAPE médio: {melhor}")
    return tab


# ─── ITEM 1: os dados antigos ajudam ou atrapalham o regime novo? ───────────
def comparar_origem_treino(df, treinar_e_prever, meses_teste=3):
    df = df.copy()
    df['_data'] = pd.to_datetime(df['data_transacao'])
    novo = df[classificar_regime(df['_data']) == 'novo'].sort_values('_data')
    corte = novo['_data'].max() - pd.DateOffset(months=meses_teste)
    te = novo[novo['_data'] > corte]
    tr_novo = novo[novo['_data'] <= corte]
    tr_tudo = df[df['_data'] <= corte]

    print("\n" + "=" * 80)
    print("REGIME NOVO — origem do treino: SÓ NOVO vs TUDO desde 2008")
    print("=" * 80)
    print(f"  teste fixo: {len(te):,} linhas (> {corte.date()})")
    out = {}
    for rotulo, tr in [('SÓ regime novo', tr_novo), ('TUDO desde 2008', tr_tudo)]:
        m = calcular_metricas(te[TARGET].values, treinar_e_prever(tr, te))
        out[rotulo] = m
        print(f"  {rotulo} ({len(tr):,}): MAPE {m['MAPE']:.1f}% | "
              f"MdAPE {m['MdAPE']:.1f}% | R² {m['R²']:.3f}")
    d = out['TUDO desde 2008']['MdAPE'] - out['SÓ regime novo']['MdAPE']
    print(f"\n  → treinar com tudo {'ATRAPALHA' if d > 0 else 'AJUDA'} "
          f"(ΔMdAPE {d:+.1f} pp).")
    return out


# ─── avaliação dos .pkl salvos (compat com seu evaluate.py antigo) ──────────
def avaliar_modelos_teste(X_test, y_test):
    print("=" * 80)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE (modelos salvos)")
    print("=" * 80)
    resultados = []
    for nome_arquivo in ['random_forest.pkl', 'xgboost.pkl', 'lightgbm.pkl']:
        caminho = OUTPUTS_MODELS / nome_arquivo
        if not caminho.exists():
            continue
        with open(caminho, 'rb') as f:
            modelo = pickle.load(f)
        nome = nome_arquivo.replace('.pkl', '').replace('_', ' ').title()
        y_pred = np.expm1(modelo.predict(X_test[modelo.feature_names_in_]))
        resultados.append(calcular_metricas(y_test, y_pred, nome))
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUTPUTS_TABLES / 'resultados_teste.csv', index=False)
    print(df_res.to_string(index=False))
    return df_res


if __name__ == '__main__':
    df = pd.read_csv(ITBI_FINAL)
    anos = list(range(2018, 2024))

    # TODOS os modelos no MESMO backtest, para a comparação ser completa.
    modelos = {
        'Baseline R$/m²': fn_baseline,
        'XGBoost': fn_xgboost,
        'LightGBM': fn_lightgbm,
        'Ensemble': fn_ensemble,
    }

    # (1) Série inteira — mostra o salto de regime e compara todos.
    tab_serie = tabela_comparativa(df, modelos, anos, apenas_regime_novo=False)

    # (2) Regime novo — a régua honesta para escolher o modelo.
    tab_novo = tabela_comparativa(df, modelos, anos, apenas_regime_novo=True)

    # (3) Salva as tabelas para o artigo.
    tab_serie.to_csv(OUTPUTS_TABLES / 'comparativo_serie_inteira.csv', index=False)
    tab_novo.to_csv(OUTPUTS_TABLES / 'comparativo_regime_novo.csv', index=False)

    # (4) Os dados antigos ajudam o regime novo? (com o modelo campeão)
    comparar_origem_treino(df, fn_lightgbm)

    print("\nTabelas comparativas salvas em outputs/tables/.")