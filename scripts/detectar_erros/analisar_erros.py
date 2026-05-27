import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, COLUNAS_EXCLUIR_MODELO, TARGET

# Carregar modelo e dados
print("Carregando modelo e dados...")
with open(OUTPUTS_MODELS / 'xgboost.pkl', 'rb') as f:
    modelo = pickle.load(f)

df = pd.read_csv(ITBI_FINAL)
df_teste = df[df['ano_transacao'] >= 2023].copy()

print(f"Imóveis de teste (2023-2024): {len(df_teste):,}\n")

# Preparar features

X_teste = df_teste.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                                 if c in df_teste.columns])
y_teste = df_teste[TARGET]

# Fazer previsões
y_pred = np.expm1(modelo.predict(X_teste))

# Calcular erros
erros_abs = np.abs(y_teste - y_pred)
erros_pct = np.abs((y_teste - y_pred) / y_teste) * 100

# Adicionar ao dataframe
df_teste['previsto'] = y_pred
df_teste['erro_abs'] = erros_abs
df_teste['erro_pct'] = erros_pct

print("="*80)
print("ANÁLISE DE ERROS")
print("="*80)

# 1. Estatísticas gerais
print(f"\n1. ESTATÍSTICAS GERAIS:")
print(f"Erro médio absoluto: R$ {erros_abs.mean():>12,.2f}")
print(f"Erro mediano absoluto: R$ {erros_abs.median():>12,.2f}")
print(f"Erro médio percentual: {erros_pct.mean():>12.2f}%")
print(f"Erro mediano percentual: {erros_pct.median():>12.2f}%")

# 2. Distribuição de valores
print(f"\n2.DISTRIBUIÇÃO DE VALORES REAIS:")
print(f"Mínimo: R$ {y_teste.min():>12,.2f}")
print(f"Q1: R$ {y_teste.quantile(0.25):>12,.2f}")
print(f"Mediana: R$ {y_teste.median():>12,.2f}")
print(f"Q3: R$ {y_teste.quantile(0.75):>12,.2f}")
print(f"Máximo: R$ {y_teste.max():>12,.2f}")

# 3. Imóveis muito baratos (< R$ 100k) — MAPE explode aqui!
baratos = df_teste[df_teste['valor_base_calculo_real'] < 100000]
print(f"\n3. IMÓVEIS MUITO BARATOS (< R$ 100k):")
print(f"Quantidade: {len(baratos):,} ({len(baratos)/len(df_teste)*100:.1f}%)")
if len(baratos) > 0:
    print(f"MAPE médio: {baratos['erro_pct'].mean():.2f}%")
    print(f"MAE médio: R$ {baratos['erro_abs'].mean():,.2f}")

# 4. Imóveis caros (> R$ 1M)
caros = df_teste[df_teste['valor_base_calculo_real'] > 1000000]
print(f"\n4. IMÓVEIS CAROS (> R$ 1M):")
print(f"Quantidade: {len(caros):,} ({len(caros)/len(df_teste)*100:.1f}%)")
if len(caros) > 0:
    print(f"MAPE médio: {caros['erro_pct'].mean():.2f}%")
    print(f"MAE médio: R$ {caros['erro_abs'].mean():,.2f}")

# 5. Imóveis "normais" (R$ 100k - R$ 1M)
normais = df_teste[(df_teste['valor_base_calculo_real'] >= 100000) &
                    (df_teste['valor_base_calculo_real'] <= 1000000)]
print(f"\n5.IMÓVEIS 'NORMAIS' (R$ 100k - R$ 1M):")
print(f"Quantidade: {len(normais):,} ({len(normais)/len(df_teste)*100:.1f}%)")
print(f"MAPE médio: {normais['erro_pct'].mean():.2f}%")
print(f"MAE médio: R$ {normais['erro_abs'].mean():,.2f}")

# 6. Top 10 piores previsões
print(f"\n6. TOP 10 PIORES PREVISÕES (erro percentual):")
piores = df_teste.nlargest(10, 'erro_pct')[
    ['bairro', 'area_construida_m2', 'idade_imovel', 'valor_base_calculo_real', 'previsto', 'erro_pct']
]
for idx, row in piores.iterrows():
    print(f"\n {row['bairro']:20s} | {row['area_construida_m2']:>6.0f}m² | {row['idade_imovel']:>2.0f} anos")
    print(f"Real: R$ {row['valor_base_calculo_real']:>12,.2f} | Previsto: R$ {row['previsto']:>12,.2f}")
    print(f"Erro: {row['erro_pct']:>6.1f}%")

# 7. Calcular MAPE apenas para imóveis "normais"
mape_normais = normais['erro_pct'].mean()
print(f"\n" + "="*80)
print(f"CONCLUSÃO:")
print(f"="*80)
print(f"\nMAPE geral (todos imóveis): {erros_pct.mean():.2f}%")
print(f"MAPE filtrado (R$ 100k - R$ 1M): {mape_normais:.2f}%")
