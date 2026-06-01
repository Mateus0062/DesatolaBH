import sys
import pandas as pd
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

Geooficados = Path(__file__).resolve().parent.parent.parent / 'data' / 'external'

geo = pd.read_csv(Geooficados / "ceps_geocodificados1.csv", dtype={'cep': str})
print(geo['tipo_resultado'].value_counts())
print(geo['precisao'].value_counts())