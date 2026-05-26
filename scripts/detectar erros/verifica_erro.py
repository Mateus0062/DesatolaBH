import pandas as pd
from config import ITBI_FINAL
df = pd.read_csv(ITBI_FINAL)
print("Colunas string:", df.select_dtypes(include='object').columns.tolist())
print("Total de colunas:", len(df.columns))