import pandas as pd
import numpy as np
from pathlib import Path
import sys

from src.feature_engineering.features_espaciais import criar_features_espaciais

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_CLEANED, ITBI_FINAL, TARGET


# =============================================================================
#  ITEM 4 — Estatísticas de bairro com JANELA RETROATIVA (backward).
#
#  Antes: janela fixa 2008-2022 + LOO. Uma transação de 2009 "via" 2010-2022 (o
#  próprio futuro) -> a feature era muito mais informativa no treino que no
#  teste, inflando o ajuste in-sample e o CV do tuning.
#  Agora: cada linha do ano Y só usa anos < Y (expanding). Treino e teste ficam
#  simétricos (ambos só olham o passado). Mantemos os MESMOS nomes de coluna
#  (preco_medio_bairro_loo, etc.) para nada quebrar a jusante — o sufixo 'loo'
#  agora é legado.
# =============================================================================
def _stats_bairro_backward(df, target=TARGET, min_n=30):
    """Devolve um DataFrame indexado como df com mean/std/count/min/max/pm2 do
    bairro calculados SÓ sobre anos anteriores ao de cada linha."""
    df = df.sort_values('ano_transacao').copy()
    df['_t2'] = df[target] ** 2

    g = df.groupby(['bairro', 'ano_transacao']).agg(
        soma=(target, 'sum'),
        soma2=('_t2', 'sum'),
        n=(target, 'size'),
        minimo=(target, 'min'),
        maximo=(target, 'max'),
        soma_area=('area_construida_m2', 'sum'),
    ).sort_index()

    cum = g.groupby(level='bairro').cumsum()
    cum_prev = cum.groupby(level='bairro').shift(1)
    # min/max expanding-backward (cummin/cummax dos mínimos/máximos por ano)
    cum_prev['min_bw'] = (g['minimo'].groupby(level='bairro').cummin()
                          .groupby(level='bairro').shift(1))
    cum_prev['max_bw'] = (g['maximo'].groupby(level='bairro').cummax()
                          .groupby(level='bairro').shift(1))

    cum_prev['media'] = cum_prev['soma'] / cum_prev['n']
    var = cum_prev['soma2'] / cum_prev['n'] - cum_prev['media'] ** 2
    cum_prev['std'] = np.sqrt(var.clip(lower=0))
    cum_prev['pm2'] = cum_prev['soma'] / cum_prev['soma_area']

    # fallback global backward (média do que veio antes do ano)
    glob = df.groupby('ano_transacao')[target].agg(['sum', 'count']).sort_index()
    glob_cum = glob.cumsum().shift(1)
    glob_cum['media'] = glob_cum['sum'] / glob_cum['count']
    media_glob = df['ano_transacao'].map(glob_cum['media']).fillna(df[target].mean())

    chave = list(zip(df['bairro'], df['ano_transacao']))
    n_bw = pd.Series([cum_prev['n'].get(k, np.nan) for k in chave], index=df.index)
    confiavel = n_bw >= min_n

    def _col(nome, fallback):
        s = pd.Series([cum_prev[nome].get(k, np.nan) for k in chave], index=df.index)
        s = s.where(confiavel, fallback)
        return s.replace([np.inf, -np.inf], np.nan).fillna(fallback)

    out = pd.DataFrame(index=df.index)
    out['media'] = _col('media', media_glob)
    out['std'] = _col('std', df[target].std())
    out['count'] = n_bw.fillna(0)
    out['minimo'] = _col('min_bw', media_glob)
    out['maximo'] = _col('max_bw', media_glob)
    pm2_glob = df[target].sum() / df['area_construida_m2'].sum()
    out['pm2'] = _col('pm2', pm2_glob)
    return out.reindex(df.index)


def criar_features_proporcoes(df, verbose=True):
    print("\n[1/9] Criando features de proporções...")
    df['razao_area_util'] = df['area_construida_m2'] / df['area_terreno_m2']
    print("1 feature de proporção criada")
    return df


def criar_features_idade(df):
    print("\n[2/9] Criando features de idade (depreciação não-linear)...")
    idade_mediana = df['idade_imovel'].median()
    df['idade_imovel'] = df['idade_imovel'].fillna(idade_mediana)
    df['imovel_novo'] = (df['idade_imovel'] <= 5).astype(int)

    # ITEM 6: depreciação não-linear + flag de provável teardown.
    # (A linear antiga capava em 40%; fraca para casas de 50-65 anos.)
    idade = df['idade_imovel'].clip(lower=0)
    df['depreciacao_estimada'] = np.minimum(idade * 0.01, 0.40)  # legado
    df['fator_construcao'] = 0.25 + 0.75 * np.exp(-0.015 * idade)
    df['provavel_teardown'] = (idade >= 40).astype(int)
    print("4 features de idade criadas")
    return df


def criar_features_interacao_simples(df):
    print("\n[3/9] Criando features de interação...")
    df['area_x_idade'] = df['area_construida_m2'] * df['idade_imovel']
    area_mediana = df['area_construida_m2'].median()
    df['novo_e_grande'] = (
        (df['idade_imovel'] <= 10) & (df['area_construida_m2'] > area_mediana)
    ).astype(int)
    print("2 features de interação criadas")
    return df


def criar_features_bairro_sem_leakage(df):
    print("\n[4/9] Calculando features de bairro (janela retroativa)...")
    s = _stats_bairro_backward(df)
    df['preco_medio_bairro_loo'] = s['media']      # nome legado (agora backward)
    df['std_preco_bairro'] = s['std']
    df['num_transacoes_bairro'] = s['count']
    df['preco_min_bairro'] = s['minimo']
    df['preco_max_bairro'] = s['maximo']
    df['range_preco_bairro'] = s['maximo'] - s['minimo']
    print("6 features de bairro criadas (sem leakage temporal)")
    return df


def criar_features_preco_m2_bairro_sem_leakage(df):
    print("\n[5/9] Calculando preço/m² médio do bairro (janela retroativa)...")
    s = _stats_bairro_backward(df)
    df['preco_m2_medio_bairro_loo'] = s['pm2']
    print("Feature criada: preco_m2_medio_bairro_loo")
    return df


def criar_features_valorizacao_bairro(df):
    print("\n[6/9] Calculando valorização do bairro (janela retroativa)...")
    yr_mean = (df.groupby(['bairro', 'ano_transacao'])[TARGET].mean())
    tabela = {}
    for bairro, sub in yr_mean.groupby(level='bairro'):
        anos = sub.index.get_level_values('ano_transacao').tolist()
        m_por_ano = dict(zip(anos, sub.values))
        for Y in anos:
            ant = sorted(a for a in anos if a < Y)  # só anos anteriores
            if len(ant) >= 1:
                ref_rec = m_por_ano[ant[-1]]
                ref_ant = m_por_ano[ant[-4]] if len(ant) >= 4 else m_por_ano[ant[0]]
                val = ((ref_rec - ref_ant) / ref_ant * 100) if ref_ant > 0 else 0.0
            else:
                val = 0.0
            tabela[(bairro, Y)] = val
    chave = list(zip(df['bairro'], df['ano_transacao']))
    df['valorizacao_bairro_3anos'] = pd.Series(
        [tabela.get(k, 0.0) for k in chave], index=df.index)
    print("Feature criada: valorizacao_bairro_3anos")
    return df


def criar_features_comparativas(df):
    print("\n[7/9] Criando features comparativas (janela retroativa)...")
    # médias de área/idade do bairro só com anos anteriores
    df = df.sort_values('ano_transacao').copy()
    g_area = df.groupby(['bairro', 'ano_transacao'])['area_construida_m2'] \
        .agg(['sum', 'count']).sort_index()
    cum = g_area.groupby(level='bairro').cumsum().groupby(level='bairro').shift(1)
    media_area_bw = cum['sum'] / cum['count']
    g_idade = df.groupby(['bairro', 'ano_transacao'])['idade_imovel'] \
        .agg(['sum', 'count']).sort_index()
    cum_i = g_idade.groupby(level='bairro').cumsum().groupby(level='bairro').shift(1)
    media_idade_bw = cum_i['sum'] / cum_i['count']

    chave = list(zip(df['bairro'], df['ano_transacao']))
    area_global = df['area_construida_m2'].mean()
    idade_global = df['idade_imovel'].mean()
    area_ref = pd.Series([media_area_bw.get(k, np.nan) for k in chave],
                         index=df.index).fillna(area_global)
    idade_ref = pd.Series([media_idade_bw.get(k, np.nan) for k in chave],
                          index=df.index).fillna(idade_global)

    df['area_vs_media_bairro'] = df['area_construida_m2'] / area_ref
    df['idade_vs_media_bairro'] = df['idade_imovel'] / (idade_ref + 1)
    print("2 features comparativas criadas")
    return df


# =============================================================================
#  ITEM 6 — Comparáveis espaço-temporais: R$/m² mediano de transações passadas
#  parecidas (mesmo bairro+tipo, área dentro de ±tol), de anos anteriores.
# =============================================================================
def criar_features_comparaveis(df, tol_area=0.30, min_comp=5):
    print("\n[8/9] Criando comparáveis (R$/m² de vizinhos parecidos)...")
    df = df.sort_values(['ano_transacao', 'mes_transacao']).copy()
    df['_pm2'] = df[TARGET] / df['area_construida_m2']
    df['comparaveis_pm2'] = np.nan
    pm2_global = df['_pm2'].median()

    for _, sub in df.groupby([df['bairro'].astype(str),
                              df['tipo_construtivo'].astype(str)]):
        sub = sub.sort_values('ano_transacao')
        anos = sub['ano_transacao'].values
        areas = sub['area_construida_m2'].values
        pm2 = sub['_pm2'].values
        idx = sub.index.values
        vals = np.full(len(sub), np.nan)
        for i in range(len(sub)):
            mask = (anos < anos[i]) & (np.abs(areas - areas[i]) <= tol_area * areas[i])
            if mask.sum() >= min_comp:
                vals[i] = np.median(pm2[mask])
        df.loc[idx, 'comparaveis_pm2'] = vals

    df['comparaveis_pm2'] = df['comparaveis_pm2'].fillna(pm2_global)
    df.drop(columns=['_pm2'], inplace=True)
    print("1 feature de comparáveis criada")
    return df


# =============================================================================
#  ITEM 3 — zona_uso como TARGET ENCODING TEMPORAL (numérico, sem leakage).
#  A string 'zona_uso' continua excluída no config; 'zona_uso_te' é que entra.
# =============================================================================
def criar_feature_zona_uso(df, coluna='zona_uso', suavizacao=50):
    print("\n[9/9] Target encoding temporal de zona_uso...")
    if coluna not in df.columns:
        print(f"  [aviso] '{coluna}' ausente — pulando.")
        return df
    df = df.sort_values('ano_transacao').copy()

    glob = df.groupby('ano_transacao')[TARGET].agg(['sum', 'count']).sort_index()
    glob_cum = glob.cumsum().shift(1)
    glob_cum['media'] = glob_cum['sum'] / glob_cum['count']
    media_glob_bw = df['ano_transacao'].map(glob_cum['media']).fillna(df[TARGET].mean())

    g = df.groupby([coluna, 'ano_transacao'])[TARGET].agg(['sum', 'count']).sort_index()
    cum = g.groupby(level=0).cumsum()
    cum_prev = cum.groupby(level=0).shift(1)
    chave = list(zip(df[coluna], df['ano_transacao']))
    soma = pd.Series([cum_prev['sum'].get(k, np.nan) for k in chave], index=df.index)
    cont = pd.Series([cum_prev['count'].get(k, np.nan) for k in chave], index=df.index)

    enc = (soma + suavizacao * media_glob_bw) / (cont + suavizacao)
    df['zona_uso_te'] = enc.fillna(media_glob_bw)
    print("Feature criada: zona_uso_te")
    return df


def criar_features_sazonalidade(df):
    print("\n[+] Criando features de sazonalidade...")
    df['trimestre'] = df['mes_transacao'].apply(lambda m: (m - 1) // 3 + 1)
    df['fim_de_ano'] = (df['mes_transacao'] >= 11).astype(int)
    df['inicio_ano'] = (df['mes_transacao'] <= 3).astype(int)
    print("3 features de sazonalidade criadas")
    return df


def aplicar_features_completo(df_input=None, path_input=None, salvar=True):
    if df_input is None:
        if path_input is None:
            path_input = ITBI_CLEANED
        df = pd.read_csv(path_input)
    else:
        df = df_input.copy()

    print("=" * 80)
    print("FEATURE ENGINEERING SEM DATA LEAKAGE (janela retroativa)")
    print("=" * 80)
    print(f"\nLinhas de entrada: {len(df):,}")
    print(f"Features iniciais: {len(df.columns)}")

    df = criar_features_proporcoes(df)
    df = criar_features_idade(df)
    df = criar_features_interacao_simples(df)
    df = criar_features_bairro_sem_leakage(df)
    df = criar_features_preco_m2_bairro_sem_leakage(df)
    df = criar_features_valorizacao_bairro(df)
    df = criar_features_espaciais(df)
    df = criar_features_comparativas(df)
    df = criar_features_comparaveis(df)
    df = criar_feature_zona_uso(df)
    df = criar_features_sazonalidade(df)

    print("\nFEATURE ENGINEERING CONCLUÍDO")
    print(f"\nFeatures finais: {len(df.columns)}")

    if salvar:
        df.to_csv(ITBI_FINAL, index=False, encoding='utf-8')
        print(f"\n✓ Dataset salvo: {ITBI_FINAL}")
        print(f"Tamanho: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    return df


if __name__ == '__main__':
    print("Aplicando feature engineering...")
    df_final = aplicar_features_completo()
    print("\nCONCLUÍDO")