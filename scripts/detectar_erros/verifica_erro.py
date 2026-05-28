"""
scripts/detectar_erros/diagnostico_leakage_overfitting.py

Bateria completa de diagnóstico, em duas frentes:

  PARTE A — LEAKAGE (sobre os modelos JÁ treinados, sem retreinar)
    A1. Confirma que nenhuma versão do alvo está em X.
    A2. Correlação de cada feature com o alvo (ranking).
    A3. Teste de leakage temporal das features de bairro: recalcula as
        estatísticas de bairro usando SÓ o treino e mede o quanto elas
        mudam em relação às do dataset inteiro.

  PARTE B — OVERFITTING (sobre os modelos JÁ treinados)
    B1. Gap dev-vs-teste por modelo (MAPE, mediana, R²).
    B2. Erro por faixa de preço (onde mora o overfit).
    B3. Importância das features no modelo campeão (concentração).

  PARTE C — TESTE QUE EXIGE RETREINO (opcional, controlado por flag)
    C1. Remove as features de bairro derivadas de preço, retreina o
        campeão e mede quanto o teste piora — quantifica a dependência
        do modelo dessas features.

Carrega os modelos de outputs/models/ (pickle), igual ao evaluate.py.
Não altera nada no pipeline; só mede.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, TARGET, COLUNAS_EXCLUIR_MODELO

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
RODAR_PARTE_C = False   # True para rodar o teste que retreina (mais lento)
CAMPEAO = 'xgboost'   # modelo para análises B3 e C

# Features de bairro derivadas de preço (alvo dos testes de leakage)
FEATURES_BAIRRO_PRECO = [
    'preco_medio_bairro_loo', 'preco_m2_medio_bairro_loo',
    'std_preco_bairro', 'preco_min_bairro', 'preco_max_bairro',
    'range_preco_bairro', 'valorizacao_bairro_3anos',
    'area_vs_media_bairro', 'idade_vs_media_bairro', 'preco_vs_tipo',
]


def carregar_modelo(nome):
    caminho = OUTPUTS_MODELS / f'{nome}.pkl'
    with open(caminho, 'rb') as f:
        return pickle.load(f)


def preparar_X_y(df):
    y = df[TARGET].copy()
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                         if c in df.columns])
    return X, y


def metricas(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    erro_pct = np.abs((y_true - y_pred) / y_true) * 100
    return {
        'MAPE': erro_pct.mean(),
        'Mediana_%': np.median(erro_pct),
        'R2': 1 - np.sum((y_true - y_pred) ** 2) /
                  np.sum((y_true - y_true.mean()) ** 2),
    }


# ============================================================================
print("=" * 80)
print("DIAGNÓSTICO DE LEAKAGE E OVERFITTING")
print("=" * 80)

df = pd.read_csv(ITBI_FINAL)
print(f"\n{len(df):,} linhas carregadas")

anos = df['ano_transacao'].values
mask_dev = anos <= 2022
mask_test = anos >= 2023

X, y = preparar_X_y(df)
X_dev, y_dev = X[mask_dev], y[mask_dev]
X_test, y_test = X[mask_test], y[mask_test]


# ============================================================================
# PARTE A — LEAKAGE
# ============================================================================
print("\n" + "=" * 80)
print("PARTE A — LEAKAGE")
print("=" * 80)

# A1 — Nenhuma versão do alvo em X
print("\n[A1] Versões do alvo presentes em X (deveria ser vazio):")
suspeitas_alvo = ['valor_declarado', 'valor_declarado_real',
                  'valor_base_calculo', 'valor_base_calculo_real',
                  'preco_m2']
presentes = [c for c in suspeitas_alvo if c in X.columns]
print(f"   {presentes if presentes else '(nenhuma — OK)'}")

# A2 — Correlação de cada feature com o alvo
print("\n[A2] Correlação absoluta de cada feature com o alvo (top 15):")
corr = X.corrwith(y).abs().sort_values(ascending=False)
for feat, val in corr.head(15).items():
    flag = "  <-- SUSPEITO" if val > 0.95 else ""
    print(f"   {feat:32s} {val:.4f}{flag}")
maxc = corr.max()
print(f"\n   Correlação máxima: {maxc:.4f} "
      f"({'sem leakage óbvio' if maxc < 0.95 else 'INVESTIGAR'})")

# A3 — Leakage temporal das features de bairro
print("\n[A3] Leakage temporal das features de bairro:")
print("     Recalcula preco_medio_bairro_loo usando SÓ treino (<=2022)")
print("     e compara com o valor atual (calculado sobre todos os anos).")

# Recalcula a média de bairro LOO usando só treino como base
df_dev = df[mask_dev]
soma_dev = df_dev.groupby('bairro')[TARGET].sum()
count_dev = df_dev.groupby('bairro')[TARGET].count()
media_geral_dev = df_dev[TARGET].mean()

# Para imóveis de teste: média do bairro calculada só com treino (sem LOO,
# pois o imóvel de teste não está na base de treino)
media_bairro_dev = (soma_dev / count_dev)
df_test_local = df[mask_test].copy()
media_so_treino = df_test_local['bairro'].map(media_bairro_dev).fillna(media_geral_dev)

if 'preco_medio_bairro_loo' in df.columns:
    media_atual_teste = df_test_local['preco_medio_bairro_loo']
    dif_rel = ((media_atual_teste - media_so_treino).abs() /
               media_so_treino.replace(0, np.nan)).median()
    print(f"     Diferença mediana entre as duas formas de calcular: "
          f"{dif_rel * 100:.1f}%")
    print(f"     (diferença alta = a feature atual usa info de 2023-2024 "
          f"que não estaria disponível num cenário honesto)")
else:
    print("     preco_medio_bairro_loo não encontrado no dataset.")


# ============================================================================
# PARTE B — OVERFITTING
# ============================================================================
print("\n" + "=" * 80)
print("PARTE B — OVERFITTING")
print("=" * 80)

modelos = {}
for nome in ['random_forest', 'xgboost', 'lightgbm']:
    try:
        modelos[nome] = carregar_modelo(nome)
    except FileNotFoundError:
        print(f"   [aviso] {nome}.pkl não encontrado, pulando.")

# B1 — Gap dev vs teste
print("\n[B1] Gap desenvolvimento vs teste (modelos treinados em log):")
print(f"   {'Modelo':16s} {'MAPE_dev':>9s} {'MAPE_test':>10s} "
      f"{'gap':>7s} {'Med_dev':>8s} {'Med_test':>9s}")
for nome, modelo in modelos.items():
    pred_dev = np.expm1(modelo.predict(X_dev))
    pred_test = np.expm1(modelo.predict(X_test))
    m_dev = metricas(y_dev, pred_dev)
    m_test = metricas(y_test, pred_test)
    gap = m_test['MAPE'] - m_dev['MAPE']
    print(f"   {nome:16s} {m_dev['MAPE']:>8.2f}% {m_test['MAPE']:>9.2f}% "
          f"{gap:>6.2f}% {m_dev['Mediana_%']:>7.2f}% "
          f"{m_test['Mediana_%']:>8.2f}%")

# B2 — Erro por faixa de preço (no campeão)
print(f"\n[B2] Erro por faixa de preço — modelo {CAMPEAO} (no teste):")
modelo_c = modelos[CAMPEAO]
pred_test_c = np.expm1(modelo_c.predict(X_test))
y_test_arr = np.asarray(y_test, float)
erro_pct = np.abs((y_test_arr - pred_test_c) / y_test_arr) * 100

faixas = [(0, 100_000), (100_000, 300_000), (300_000, 600_000),
          (600_000, 1_000_000), (1_000_000, np.inf)]
print(f"   {'Faixa (R$)':24s} {'n':>7s} {'MAPE':>8s} {'Mediana':>9s}")
for lo, hi in faixas:
    m = (y_test_arr >= lo) & (y_test_arr < hi)
    if m.sum() == 0:
        continue
    rotulo = f"{lo/1000:.0f}k - {hi/1000:.0f}k" if hi != np.inf else f">{lo/1000:.0f}k"
    print(f"   {rotulo:24s} {m.sum():>7,} {erro_pct[m].mean():>7.2f}% "
          f"{np.median(erro_pct[m]):>8.2f}%")

# B3 — Importância das features no campeão
print(f"\n[B3] Importância das features — {CAMPEAO} (top 12):")
if hasattr(modelo_c, 'feature_importances_'):
    imp = pd.Series(modelo_c.feature_importances_,
                    index=modelo_c.feature_names_in_).sort_values(ascending=False)
    for feat, val in imp.head(12).items():
        print(f"   {feat:32s} {val*100:>6.2f}%")
    print(f"\n   Top-1 concentra {imp.iloc[0]*100:.1f}% | "
          f"Top-3 concentram {imp.head(3).sum()*100:.1f}%")
    imp_bairro = imp[[f for f in FEATURES_BAIRRO_PRECO
                      if f in imp.index]].sum()
    print(f"   Features de bairro derivadas de preço somam "
          f"{imp_bairro*100:.1f}% da importância")


# ============================================================================
# PARTE C — TESTE QUE EXIGE RETREINO (opcional)
# ============================================================================
if RODAR_PARTE_C:
    print("\n" + "=" * 80)
    print("PARTE C — IMPACTO DAS FEATURES DE BAIRRO (retreina o campeão)")
    print("=" * 80)
    from sklearn.ensemble import RandomForestRegressor
    from config import HIPERPARAMETROS

    feats_remover = [f for f in FEATURES_BAIRRO_PRECO if f in X.columns]
    print(f"\n   Removendo {len(feats_remover)} features de bairro: "
          f"{feats_remover}")

    X_dev_sem = X_dev.drop(columns=feats_remover)
    X_test_sem = X_test.drop(columns=feats_remover)

    modelo_sem = RandomForestRegressor(
        **HIPERPARAMETROS['random_forest'], random_state=42, n_jobs=-1)
    modelo_sem.fit(X_dev_sem, np.log1p(y_dev))
    pred_sem = np.expm1(modelo_sem.predict(X_test_sem))
    m_sem = metricas(y_test, pred_sem)

    m_com = metricas(y_test, pred_test_c)
    print(f"\n   COM features de bairro:  MAPE {m_com['MAPE']:.2f}%, "
          f"R² {m_com['R2']:.4f}")
    print(f"   SEM features de bairro:  MAPE {m_sem['MAPE']:.2f}%, "
          f"R² {m_sem['R2']:.4f}")
    print(f"   Degradação: MAPE +{m_sem['MAPE']-m_com['MAPE']:.2f}pp, "
          f"R² {m_sem['R2']-m_com['R2']:+.4f}")
    print("\n   Interpretação: se a degradação for grande, o modelo depende "
          "fortemente\n   dessas features (frágil); se pequena, elas são "
          "redundantes com área/local.")
else:
    print("\n[PARTE C pulada — defina RODAR_PARTE_C=True para o teste de "
          "remoção de features]")

print("\n" + "=" * 80)
print("DIAGNÓSTICO CONCLUÍDO")
print("=" * 80)