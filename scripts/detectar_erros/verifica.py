import pandas as pd
from config import ITBI_FINAL
df = pd.read_csv(ITBI_FINAL)
dev = df[df['ano_transacao'] <= 2022]
test = df[df['ano_transacao'] >= 2023]
feats = ['preco_medio_bairro_loo', 'preco_m2_medio_bairro_loo', 'std_preco_bairro',
         'area_vs_media_bairro', 'preco_vs_tipo']
for f in feats:
    print(f"\n{f}:")
    print(f"  DEV : média {dev[f].mean():.2f}, mediana {dev[f].median():.2f}")
    print(f"  TEST: média {test[f].mean():.2f}, mediana {test[f].median():.2f}")