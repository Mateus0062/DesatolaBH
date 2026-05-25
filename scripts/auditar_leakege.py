import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ITBI_FINAL, COLUNAS_EXCLUIR_MODELO

print("=" * 80)
print("AUDITORIA DE DATA LEAKAGE")
print("=" * 80)

df = pd.read_csv(ITBI_FINAL)
y = df['valor_declarado']

# 1. Quais colunas proibidas ainda estão no dataset?
print("\n[1] Colunas proibidas presentes no dataset final:")
presentes = [c for c in COLUNAS_EXCLUIR_MODELO if c in df.columns]
for c in presentes:
    print(f"   - {c}")
if not presentes:
    print("   (nenhuma — ok)")

# 2. Correlação de TODAS as colunas numéricas com o target.
#    Qualquer coisa com |corr| > 0.95 que não seja o próprio target
#    é suspeita de leakage.
print("\n[2] Correlação com valor_declarado (|corr| > 0.80):")
num = df.select_dtypes(include=[np.number])
corr = num.corrwith(y).abs().sort_values(ascending=False)
suspeitas = corr[(corr > 0.80) & (corr.index != 'valor_declarado')]
for feat, val in suspeitas.items():
    flag = "  <-- LEAKAGE PROVÁVEL" if val > 0.95 else ""
    print(f"   {feat:35s} {val:.4f}{flag}")
if suspeitas.empty:
    print("   (nenhuma feature com correlação alta — bom sinal)")

# 3. Colunas string que sobraram
print("\n[3] Colunas não-numéricas no dataset:")
strings = df.select_dtypes(include='object').columns.tolist()
print(f" {strings if strings else '(nenhuma)'}")

print("\n" + "=" * 80)
print("INTERPRETAÇÃO:")
print("  - valor_base_calculo costuma ter corr > 0.98 com o target.")
print("    Se ele aparece em [2], NUNCA pode ir ao modelo.")
print("  - Se você já treinou com ele incluso, seu R² está inflado.")
print("=" * 80)