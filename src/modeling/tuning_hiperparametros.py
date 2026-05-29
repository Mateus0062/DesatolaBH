import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, OUTPUTS_TABLES, TARGET, COLUNAS_EXCLUIR_MODELO,
                    MONOTONIC_CONSTRAINTS, PREVER_PRECO_M2)

N_ITER = 60                # combinações testadas por modelo (era 50)
N_FOLDS = 4
RANDOM_STATE = 42
MODELOS_A_TUNAR = ['xgboost', 'lightgbm', 'random_forest']

# Teto alto de árvores: no train.py o early stopping bateu em 2000, sinal de que
# os boosters queriam mais capacidade. Aqui damos folga e deixamos o early
# stopping de cada fold cortar antes.
N_ESTIMATORS_TETO = 4000
EARLY_STOPPING_ROUNDS = 50


# =============================================================================
#  UTILITÁRIOS
# =============================================================================
def _converter(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _vetor_monotonia(colunas, framework):
    """Mesma lógica do train.py: tuning e treino otimizam o MESMO modelo."""
    cons = []
    for c in colunas:
        s = MONOTONIC_CONSTRAINTS.get(c, 0)
        if PREVER_PRECO_M2 and c in ('area_construida_m2', 'area_total_m2',
                                     'area_terreno_m2'):
            s = 0
        cons.append(s)
    if framework == 'xgboost':
        return '(' + ','.join(str(x) for x in cons) + ')'
    return cons


# Scorer: MdAPE no espaço REAL (R$). O modelo treina em log; aqui revertimos
# com expm1 e medimos o erro percentual mediano, que é a métrica do artigo e
# robusta à cauda de imóveis baratos. RandomizedSearchCV maximiza, então
# devolvemos o NEGATIVO.
def _mdape_real_scorer(estimator, X, y_log):
    y_pred = np.expm1(estimator.predict(X))
    y_true = np.expm1(np.asarray(y_log, float))
    if PREVER_PRECO_M2:
        # y_log está em log(R$/m²): converte ambos para R$/m², compara em %
        pass  # a razão percentual é invariante à multiplicação por área
    ape = np.abs((y_true - y_pred) / y_true) * 100
    return -np.median(ape)


# =============================================================================
#  1. CARREGAR E PREPARAR DADOS (só treino+val: <= 2022)
# =============================================================================
print("=" * 80)
print("TUNING DE HIPERPARÂMETROS (regularizado, com early stopping no CV)")
print("=" * 80)

df = pd.read_csv(ITBI_FINAL)
print(f"\n{len(df):,} linhas carregadas")

df_tuning = df[df['ano_transacao'] <= 2022].copy()
# Ordem cronológica é essencial para o TimeSeriesSplit.
df_tuning = df_tuning.sort_values(['ano_transacao', 'mes_transacao'])
print(f"Dados de tuning (2008-2022): {len(df_tuning):,} linhas")

y_real = df_tuning[TARGET].copy()
X = df_tuning.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                            if c in df_tuning.columns])

# Travas — mesma filosofia do preparar_dados.
strings = X.select_dtypes(include='object').columns.tolist()
if strings:
    raise ValueError(f"Colunas string chegaram ao tuning: {strings}")
nans = X.columns[X.isna().any()].tolist()
if nans:
    raise ValueError(f"Features com NaN: {nans}")

# Alvo em log — mesmo espaço do train.py. Em R$/m² se o toggle estiver ligado.
if PREVER_PRECO_M2:
    y_log = np.log1p(y_real / df_tuning['area_construida_m2'])
    print("Alvo do tuning: log(R$/m²)")
else:
    y_log = np.log1p(y_real)
    print("Alvo do tuning: log(R$)")
print(f"Features: {X.shape[1]} colunas")

MONO_XGB = _vetor_monotonia(list(X.columns), 'xgboost')
MONO_LGB = _vetor_monotonia(list(X.columns), 'lightgbm')


# =============================================================================
#  2. CONFIGURAÇÃO DA BUSCA
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

# Espaços de busca com HEADROOM em relação à config regularizada atual:
# os boosters ficaram subajustados, então abrimos profundidade/aprendizado,
# mas mantendo regularização forte (alpha/lambda, subsample, colsample) para
# não voltar a decorar. n_estimators NÃO entra na busca dos boosters: usamos o
# teto + early stopping por fold.
espacos = {
    'xgboost': {
        'estimador': XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
            n_estimators=N_ESTIMATORS_TETO,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            monotone_constraints=MONO_XGB,
        ),
        'distribuicoes': {
            'max_depth': randint(4, 10),
            'learning_rate': uniform(0.01, 0.14),     # 0.01 a 0.15
            'subsample': uniform(0.6, 0.35),          # 0.60 a 0.95
            'colsample_bytree': uniform(0.6, 0.35),
            'min_child_weight': randint(5, 60),
            'reg_alpha': uniform(0.0, 3.0),
            'reg_lambda': uniform(1.0, 9.0),           # 1 a 10
            'gamma': uniform(0.0, 0.5),
        },
        'fit_params': {},  # eval_set é injetado por fold (ver _busca_boosters)
    },
    'lightgbm': {
        'estimador': LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
            n_estimators=N_ESTIMATORS_TETO,
            monotone_constraints=MONO_LGB,
        ),
        'distribuicoes': {
            'max_depth': randint(4, 12),
            'num_leaves': randint(20, 120),
            'learning_rate': uniform(0.01, 0.14),
            'subsample': uniform(0.6, 0.35),
            'subsample_freq': randint(1, 5),
            'colsample_bytree': uniform(0.6, 0.35),
            'min_child_samples': randint(20, 120),
            'reg_alpha': uniform(0.0, 3.0),
            'reg_lambda': uniform(0.0, 9.0),
        },
        'fit_params': {},
    },
    'random_forest': {
        'estimador': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        'distribuicoes': {
            'n_estimators': randint(150, 400),
            'max_depth': randint(10, 28),
            'min_samples_split': randint(5, 30),
            'min_samples_leaf': randint(3, 20),
            'max_features': uniform(0.3, 0.6),         # fração 0.3 a 0.9
        },
        'fit_params': {},
    },
}


# =============================================================================
#  3. BUSCA — boosters com early stopping por fold; RF padrão
# =============================================================================
def _busca_boosters_manual(nome, cfg):
    """
    RandomizedSearchCV não passa eval_set por fold com cronologia. Fazemos a
    busca à mão: para cada combinação, treina em cada fold do TimeSeriesSplit
    com early stopping no bloco de validação seguinte, e mede MdAPE-real.
    """
    from sklearn.model_selection import ParameterSampler
    amostrador = list(ParameterSampler(
        cfg['distribuicoes'], n_iter=N_ITER, random_state=RANDOM_STATE))
    Xv, yv = X.values, y_log.values
    melhor = None
    for i, params in enumerate(amostrador, 1):
        scores = []
        for tr_idx, va_idx in tscv.split(Xv):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = yv[tr_idx], yv[va_idx]
            mdl = cfg['estimador'].__class__(**{
                **cfg['estimador'].get_params(), **params})
            if nome == 'xgboost':
                mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:  # lightgbm
                mdl.set_params(early_stopping_round=EARLY_STOPPING_ROUNDS)
                mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                        callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS,
                                                      verbose=False)])
            y_pred = np.expm1(mdl.predict(X_va))
            y_true = np.expm1(y_va)
            ape = np.abs((y_true - y_pred) / y_true) * 100
            scores.append(np.median(ape))
        media = float(np.mean(scores))
        if melhor is None or media < melhor['score']:
            melhor = {'params': params, 'score': media}
        if i % 10 == 0 or i == 1:
            print(f"    [{i:>3}/{N_ITER}] melhor MdAPE-real até agora: "
                  f"{melhor['score']:.2f}%")
    return melhor['params'], melhor['score']


def _busca_rf(cfg):
    busca = RandomizedSearchCV(
        estimator=cfg['estimador'],
        param_distributions=cfg['distribuicoes'],
        n_iter=N_ITER, scoring=_mdape_real_scorer, cv=tscv,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1, error_score='raise',
    )
    busca.fit(X, y_log)
    return busca.best_params_, -busca.best_score_


# =============================================================================
#  4. EXECUTAR (salvamento incremental, modelo a modelo)
# =============================================================================
caminho = OUTPUTS_TABLES / 'melhores_hiperparametros.json'

for nome, cfg in espacos.items():
    if nome not in MODELOS_A_TUNAR:
        print(f"Pulando {nome} - não está em MODELOS_A_TUNAR")
        continue

    print("\n" + "=" * 80)
    print(f"TUNANDO: {nome}")
    print("=" * 80)
    print(f"{N_ITER} combinações × {N_FOLDS} folds = {N_ITER * N_FOLDS} treinos")

    inicio = time.time()
    if nome in ('xgboost', 'lightgbm'):
        melhores_params, melhor_mdape = _busca_boosters_manual(nome, cfg)
    else:
        melhores_params, melhor_mdape = _busca_rf(cfg)
    duracao = (time.time() - inicio) / 60

    print(f"\nConcluído em {duracao:.1f} min")
    print(f"Melhor MdAPE-real (CV): {melhor_mdape:.2f}%")
    print(f"Melhores parâmetros:")
    for k, v in melhores_params.items():
        print(f"  {k}: {v}")

    # Para os boosters, fixa um n_estimators razoável a partir do teto: o
    # train.py refaz o early stopping na validação, então gravamos o teto.
    params_limpos = {k: _converter(v) for k, v in melhores_params.items()}
    if nome in ('xgboost', 'lightgbm'):
        params_limpos['n_estimators'] = N_ESTIMATORS_TETO

    resultado_modelo = {
        'melhores_parametros': params_limpos,
        'melhor_mdape_real': float(melhor_mdape),
        'duracao_min': float(duracao),
    }

    if caminho.exists():
        with open(caminho, 'r', encoding='utf8') as f:
            json_final = json.load(f)
    else:
        json_final = {}
    json_final[nome] = resultado_modelo
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(json_final, f, indent=2, ensure_ascii=False)
    print(f"  → Salvo em {caminho}")

print("\n" + "=" * 80)
print("TUNING CONCLUÍDO")
print("=" * 80)
print(f"\nMelhores hiperparâmetros salvos em: {caminho}")
print("Copie cada bloco 'melhores_parametros' para HIPERPARAMETROS no config.py")
print("e rode train.py + evaluate.py de novo.")