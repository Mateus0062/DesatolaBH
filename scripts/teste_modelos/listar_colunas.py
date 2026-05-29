import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ITBI_FINAL, COLUNAS_EXCLUIR_MODELO, TARGET

print("Carregando modelo final...")
df = pd.read_csv(ITBI_FINAL)

print("\nListando todas as colunas do dataset final...\n")
for c in df.columns:
    print("", c)

print("\n============================================================")
print("Listando apenas as colunas que são usadas como features do modelo...")
df_dataset = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] if c in df.columns])
for c in df_dataset.columns:
    print(" ", c)

print("\n============================================================")
print("Correlação de cada feature com o alvo...")
df_cor = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] if c in df.columns])
Y = df[TARGET]

corr = df_cor.corrwith(Y).abs().sort_values(ascending=False)
print(corr.to_string())

print("\n==============================================================")
print("Features que tenham as menores correlações com o alvo....")
df_cor = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] if c in df.columns])
Y = df[TARGET]

corr = df_cor.corrwith(Y).abs().sort_values(ascending=True)
print(corr.to_string())


