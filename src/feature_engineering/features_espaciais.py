import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_CLEANED, TARGET

# Centro de BH (Praça Sete) — referência para distância radial.
CENTRO_BH = (-19.9191, -43.9386)

PATH_CEPS_UNICOS = Path('data/external/ceps_unicos.csv')
PATH_GEO = Path('data/external/ceps_geocodificados.csv')

# Níveis de precisão do geocodebr considerados CONFIÁVEIS (coordenada fina).
# 'municipio' e 'cep' são grosseiros (apontam para um ponto único) -> descartar
# e deixar cair no fallback de bairro, que é mais informativo.
PRECISOES_CONFIAVEIS = ('logradouro', 'numero')

# ─── passo 1: exportar CEPs únicos (+ logradouro) para o R ──────────────────
def exportar_ceps_unicos(path_input=None, path_out=PATH_CEPS_UNICOS):
    df = pd.read_csv(path_input or ITBI_CLEANED, dtype={'cep': str})
    df = df.dropna(subset=['cep'])
    df['cep'] = df['cep'].astype(str).str.zfill(8)

    # Logradouro a partir de 'endereco': pega antes do ' - ' e tira o nº final.
    df['logradouro'] = (df['endereco'].astype(str)
                        .str.split(' - ').str[0]
                        .str.replace(r'\s+\d+$', '', regex=True)
                        .str.strip())

    ceps_ruas = (df[['cep', 'logradouro']]
                 .drop_duplicates(subset=['cep'])
                 .sort_values('cep'))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    ceps_ruas.to_csv(path_out, index=False)
    print(f"[1] {len(ceps_ruas):,} CEPs únicos com logradouro -> {path_out}")
    print(f"    Agora rode 'geocodificar.R' em R para gerar {PATH_GEO.name}.")
    return ceps_ruas


# ─── passo 3: juntar coordenadas (com filtro de precisão) ───────────────────
def adicionar_coordenadas(df, path_geo=PATH_GEO,
                          precisoes_ok=PRECISOES_CONFIAVEIS):
    """Junta lat/lon por CEP, mantendo só coordenadas com precisão confiável
    (nível de logradouro/número). As demais (município/cep) caem no fallback de
    bairro, que é mais informativo que um ponto único de município."""
    if not Path(path_geo).exists():
        print(f"  [aviso] {path_geo} não encontrado — rode geocodificar.R antes. "
              "Pulando features espaciais.")
        df['lat'] = np.nan
        df['lon'] = np.nan
        return df

    geo = pd.read_csv(path_geo, dtype={'cep': str})
    geo['cep'] = geo['cep'].str.zfill(8)

    # Filtro de precisão: zera coordenadas grosseiras (município/cep).
    if 'precisao' in geo.columns:
        ruim = ~geo['precisao'].isin(precisoes_ok)
        n_ruim = int(ruim.sum())
        geo.loc[ruim, ['lat', 'lon']] = np.nan
        print(f"  {n_ruim:,} CEPs com precisão grosseira "
              f"(fora de {precisoes_ok}) -> fallback de bairro")
    else:
        print("  [aviso] coluna 'precisao' ausente no CSV — sem filtro de precisão.")

    df = df.copy()
    df['cep'] = df['cep'].astype(str).str.zfill(8)
    df = df.merge(geo[['cep', 'lat', 'lon']], on='cep', how='left')

    # Fallback 1: centroide (mediano) do bairro, calculado só com coords boas.
    falta = df['lat'].isna()
    n_direto = int((~falta).sum())
    if falta.any():
        cent = df.dropna(subset=['lat']).groupby('bairro')[['lat', 'lon']].median()
        df.loc[falta, 'lat'] = df.loc[falta, 'bairro'].map(cent['lat'])
        df.loc[falta, 'lon'] = df.loc[falta, 'bairro'].map(cent['lon'])
    # Fallback 2: centro de BH (bairro sem nenhuma coord boa).
    n_pos_bairro = int(df['lat'].isna().sum())
    df['lat'] = df['lat'].fillna(CENTRO_BH[0])
    df['lon'] = df['lon'].fillna(CENTRO_BH[1])
    print(f"  Coordenadas: {n_direto:,} diretas (logradouro) | "
          f"{int(falta.sum()) - n_pos_bairro:,} por centroide de bairro | "
          f"{n_pos_bairro:,} pelo centro de BH")
    return df


# ─── passo 4a: distância ao centro ──────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def criar_dist_centro(df):
    df['dist_centro_km'] = _haversine_km(
        df['lat'].values, df['lon'].values, CENTRO_BH[0], CENTRO_BH[1])
    print("  feature criada: dist_centro_km")
    return df


# ─── passo 4b: KNN de preço/m² dos vizinhos (sem leakage temporal) ──────────
def criar_knn_preco(df, k=10, target=TARGET):
    """Para cada imóvel do ano Y: preço/m² médio dos k vizinhos geográficos mais
    próximos com transação em anos ANTERIORES (< Y). BallTree haversine."""
    from sklearn.neighbors import BallTree
    df = df.sort_values('ano_transacao').copy()
    pm2 = (df[target] / df['area_construida_m2']).values
    coords = np.radians(df[['lat', 'lon']].values)
    df['knn_preco_m2'] = np.nan
    pm2_global = np.nanmedian(pm2)

    for Y in sorted(df['ano_transacao'].unique()):
        passado = (df['ano_transacao'] < Y).values
        atual = (df['ano_transacao'] == Y).values
        if passado.sum() < k:
            continue
        tree = BallTree(coords[passado], metric='haversine')
        _, idx = tree.query(coords[atual], k=k)
        df.loc[atual, 'knn_preco_m2'] = pm2[passado][idx].mean(axis=1)

    df['knn_preco_m2'] = df['knn_preco_m2'].fillna(pm2_global)
    print(f"  feature criada: knn_preco_m2 (k={k}, sem leakage temporal)")
    return df


def criar_features_espaciais(df, k=10, precisoes_ok=PRECISOES_CONFIAVEIS):
    print("\n[espacial] Criando features de micro-localização...")
    df = adicionar_coordenadas(df, precisoes_ok=precisoes_ok)
    df = criar_dist_centro(df)
    df = criar_knn_preco(df, k=k)
    return df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--exportar', action='store_true',
                    help='gera ceps_unicos.csv para o R')
    args = ap.parse_args()
    if args.exportar:
        exportar_ceps_unicos()
    else:
        df = pd.read_csv(ITBI_CLEANED, dtype={'cep': str})
        df = criar_features_espaciais(df)
        print(df[['cep', 'lat', 'lon', 'dist_centro_km', 'knn_preco_m2']].head())