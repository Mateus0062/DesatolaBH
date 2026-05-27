import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_TABLES, TARGET, COLUNAS_EXCLUIR_MODELO

N_ITER = 50
N_FOLDS = 4
RANDOM_STATE = 42

MODELOS_A_TUNAR = ['xgboost', 'lightgbm']

# ============================================================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================================================

print("=" * 80)
print("TUNING DE HIPERPARÂMETROS")
print("=" * 80)

df = pd.read_csv(ITBI_FINAL)
print(f"\n{len(df):,} linhas carregadas")

# Tuning roda só em treino+validação (2008-2022). Teste fica de fora.
df_tuning = df[df['ano_transacao'] <= 2022].copy()
# Ordenar por tempo é essencial: TimeSeriesSplit assume ordem cronológica.
df_tuning = df_tuning.sort_values(['ano_transacao', 'mes_transacao'])
print(f"Dados de tuning (2008-2022): {len(df_tuning):,} linhas")

y = df_tuning[TARGET].copy()
X = df_tuning.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                            if c in df_tuning.columns])

# Travas — mesma filosofia do preparar_dados.
strings = X.select_dtypes(include='object').columns.tolist()
if strings:
    raise ValueError(f"Colunas string chegaram ao tuning: {strings}")
nans = X.columns[X.isna().any()].tolist()
if nans:
    raise ValueError(f"Features com NaN: {nans}")

# Target em log — os modelos serão otimizados nesse espaço.
y_log = np.log1p(y)
print(f"Features: {X.shape[1]} colunas")

# ============================================================================
# 2. CONFIGURAÇÃO DA BUSCA
# ============================================================================

# TimeSeriesSplit: cada fold treina num período e valida no seguinte.
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

# Métrica: MAE no espaço log. Otimizar erro absoluto em log equivale,
# aproximadamente, a otimizar erro percentual no espaço real (R$).
SCORING = 'neg_mean_absolute_error'

# Espaços de busca por modelo.
# Para o RF, o teto de n_estimators é moderado para o tempo não estourar.
espacos = {
    'random_forest': {
        'estimador': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        'distribuicoes': {
            'n_estimators': randint(80, 220),
            'max_depth': randint(10, 35),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 12),
            'max_features': uniform(0.4, 0.6),  # fração: 0.4 a 1.0
        },
    },
    'xgboost': {
        'estimador': XGBRegressor(random_state=RANDOM_STATE, n_jobs=1,
                                  verbosity=0),
        'distribuicoes': {
            'n_estimators': randint(150, 600),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),  # 0.01 a 0.30
            'subsample': uniform(0.6, 0.4),        # 0.6 a 1.0
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0.0, 1.0),
            'reg_lambda': uniform(0.5, 2.0),
        },
    },
    'lightgbm': {
        'estimador': LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1,
                                   verbose=-1),
        'distribuicoes': {
            'n_estimators': randint(150, 600),
            'max_depth': randint(3, 14),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'num_leaves': randint(20, 150),
            'min_child_samples': randint(10, 80),
        },
    },
}

# ============================================================================
# 3. EXECUTAR O TUNING
# ============================================================================

resultados = {}

for nome, cfg in espacos.items():
    if nome not in MODELOS_A_TUNAR:
        print(f"Pulando {nome} - Não está em MODELOS_A_TUNAR")
        continue

    print("\n" + "=" * 80)
    print(f"TUNANDO: {nome}")
    print("=" * 80)
    print(f"  {N_ITER} combinações × {N_FOLDS} folds = {N_ITER * N_FOLDS} treinos")

    inicio = time.time()

    busca = RandomizedSearchCV(
        estimator=cfg['estimador'],
        param_distributions=cfg['distribuicoes'],
        n_iter=N_ITER,
        scoring=SCORING,
        cv=tscv,
        n_jobs=-1,            # paraleliza as combinações nos 12 núcleos
        random_state=RANDOM_STATE,
        verbose=2,
        error_score='raise',
    )

    # Treina em log — coerente com o train.py.
    busca.fit(X, y_log)

    duracao = (time.time() - inicio) / 60

    # O score é MAE negativo em log; reverter o sinal para leitura.
    melhor_mae_log = -busca.best_score_

    print(f"\nConcluído em {duracao:.1f} min")
    print(f"Melhor MAE (espaço log): {melhor_mae_log:.4f}")
    print(f"Melhores parâmetros:")
    for k, v in busca.best_params_.items():
        print(f"    {k}: {v}")

    resultados[nome] = {
        'melhores_parametros': busca.best_params_,
        'melhor_mae_log': melhor_mae_log,
        'duracao_min': duracao,
    }

# ============================================================================
# 4. SALVAR
# ============================================================================

# Converter tipos numpy para tipos Python nativos (JSON não serializa np.int).
def _converter(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

resultados_limpos = {}
for nome, r in resultados.items():
    params_limpos = {k: _converter(v) for k, v in r['melhores_parametros'].items()}
    resultados_limpos[nome] = {
        'melhores_parametros': params_limpos,
        'melhor_mae_log': float(r['melhor_mae_log']),
        'duracao_min': float(r['duracao_min']),
    }

caminho = OUTPUTS_TABLES / 'melhores_hiperparametros.json'
if caminho.exists():
    with open(caminho, 'r', encoding='utf8') as f:
        json_final = json.load(f)
    print(f"JSON existente encontrado - {list(json_final.keys())}")
else:
    json_final = {}

with open(caminho, 'w', encoding='utf-8') as f:
    json.dump(resultados_limpos, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("TUNING CONCLUÍDO")
print("=" * 80)
print(f"\nMelhores hiperparâmetros salvos em: {caminho}")