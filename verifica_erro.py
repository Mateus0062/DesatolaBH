import pandas as pd
import numpy as np
from config import ITBI_FINAL

df = pd.read_csv(ITBI_FINAL)
apt24 = df[(df['tipo_construtivo'] == 'AP') &
           (df['tipo_ocupacao'] == 'RESIDENCIAL') &
           (df['ano_transacao'] == 2024)]

for b in ['SERRA', 'SAVASSI', 'BELVEDERE', 'LOURDES']:
    s = apt24[apt24['bairro'] == b]
    if len(s):
        pm2_dec = (s['valor_declarado'] / s['area_construida_m2']).median()
        pm2_base = (s['valor_base_calculo'] / s['area_construida_m2']).median()
        print(f"{b:12s} n={len(s):>4}  "
              f"declarado=R${pm2_dec:>9,.0f}/m²  "
              f"base=R${pm2_base:>9,.0f}/m²")