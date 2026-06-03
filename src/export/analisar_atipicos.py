import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_TABLES, TARGET, OUTPUTS_MODELS_TRAIN2)
from src.modeling.evaluate import _Xy

# ── parâmetros ────────────────────────────────────────────────────────────────
MODELO_LGBM = OUTPUTS_MODELS_TRAIN2 / 'lightgbm.pkl'
ANO_TESTE_INICIO = 2023
LIMITE_ATIPICO = 100_000
MODELO_PREVE_LOG = True
OUT = OUTPUTS_TABLES / 'atipicos'
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

# colunas descritivas a incluir no relatório, SE existirem no dataset
COLS_DESCRITIVAS = [
    'bairro', 'tipo_construtivo', 'tipo_ocupacao', 'area_construida_m2',
    'area_total_m2', 'area_terreno_m2', 'idade_imovel', 'ano_construcao',
    'ano_transacao', 'mes_transacao', 'padrao_acabamento', 'cep',
    'valor_declarado_real', 'valor_base_calculo_real',
]

COR = {"normal": "#9AA0A6", "atipico": "#A32D2D", "primaria": "#2C5F8A"}


def _fmt_milhar(x, _pos=None):
    return f"{x:,.0f}".replace(",", ".")


FMT_MILHAR = mticker.FuncFormatter(_fmt_milhar)


def _carregar_modelo():
    with open(MODELO_LGBM, "rb") as f:
        return pickle.load(f)


def _prever_reais(modelo, X):
    p = np.asarray(modelo.predict(X), dtype=float)
    return np.expm1(p) if MODELO_PREVE_LOG else p


def _preparar(df):
    """Prediz no holdout e devolve o df de teste com colunas de erro."""
    teste = df[df['ano_transacao'] >= ANO_TESTE_INICIO].copy()
    X, y = _Xy(teste)
    pred = _prever_reais(modelo=_carregar_modelo(), X=X)
    real = np.asarray(y, dtype=float)
    teste['preco_real'] = real
    teste['preco_previsto'] = pred
    teste['erro_abs'] = np.abs(pred - real)
    teste['erro_pct'] = teste['erro_abs'] / np.maximum(real, 1.0) * 100
    teste['sobreavaliado'] = pred > real
    if 'area_construida_m2' in teste.columns:
        teste['preco_m2_real'] = real / teste['area_construida_m2'].replace(0, np.nan)
    return teste


def _tabela_atipicos(teste):
    atip = teste[teste['preco_real'] < LIMITE_ATIPICO].copy()
    atip = atip.sort_values('erro_pct', ascending=False)
    cols = (['preco_real', 'preco_previsto', 'erro_abs', 'erro_pct',
             'preco_m2_real', 'sobreavaliado']
            + [c for c in COLS_DESCRITIVAS if c in atip.columns])
    cols = list(dict.fromkeys(cols))  # remove duplicatas mantendo ordem
    return atip[cols]


def _salvar_tabela(atip):
    caminho = OUT / 'atipicos_tabela.csv'
    atip.to_csv(caminho, index=False, encoding='utf-8')
    print(f"  salvo: {caminho}  ({len(atip)} imóveis)")


def _relatorio_texto(teste, atip):
    """Diagnóstico em texto — comparando atípicos vs. a massa normal."""
    normal = teste[teste['preco_real'] >= LIMITE_ATIPICO]
    L = []
    L.append("=" * 70)
    L.append("RELATÓRIO — IMÓVEIS ATÍPICOS (preço real < R$ "
             f"{LIMITE_ATIPICO:,.0f})".replace(",", "."))
    L.append("=" * 70)
    L.append(f"\nTotal de atípicos no holdout (>= {ANO_TESTE_INICIO}): {len(atip)} "
             f"({len(atip)/len(teste)*100:.2f}% das {len(teste):,} transações de teste)"
             .replace(",", "."))
    L.append(f"\nERRO DO MODELO:")
    L.append(f"  Atípicos : MAPE {atip['erro_pct'].mean():6.1f}%  | "
             f"MdAPE {atip['erro_pct'].median():6.1f}%")
    L.append(f"  Normais  : MAPE {normal['erro_pct'].mean():6.1f}%  | "
             f"MdAPE {normal['erro_pct'].median():6.1f}%")
    sob = atip['sobreavaliado'].mean() * 100
    L.append(f"  O modelo SUPERAVALIA {sob:.0f}% dos atípicos "
             f"(prevê acima do valor real).")

    if 'area_construida_m2' in atip.columns:
        L.append(f"\nÁREA CONSTRUÍDA (m²):")
        L.append(f"  Atípicos : mediana {atip['area_construida_m2'].median():6.1f} | "
                 f"mín {atip['area_construida_m2'].min():.1f} | "
                 f"máx {atip['area_construida_m2'].max():.1f}")
        L.append(f"  Normais  : mediana {normal['area_construida_m2'].median():6.1f}")
    if 'preco_m2_real' in atip.columns:
        L.append(f"\nPREÇO/m² REAL (R$):")
        L.append(f"  Atípicos : mediana R$ {atip['preco_m2_real'].median():,.0f}"
                 .replace(",", "."))
        L.append(f"  Normais  : mediana R$ {normal['preco_m2_real'].median():,.0f}"
                 .replace(",", "."))
    if 'tipo_construtivo' in atip.columns:
        L.append(f"\nTIPO CONSTRUTIVO (atípicos):")
        for tipo, n in atip['tipo_construtivo'].value_counts().items():
            L.append(f"  {tipo}: {n}")
    if 'bairro' in atip.columns:
        L.append(f"\nBAIRROS mais frequentes (atípicos):")
        for b, n in atip['bairro'].value_counts().head(5).items():
            L.append(f"  {b}: {n}")

    L.append("\n" + "-" * 70)
    L.append("LEITURA (hipóteses a verificar caso a caso na tabela .csv):")
    L.append("  Áreas muito pequenas + preço/m² baixo sugerem unidades que não são")
    L.append("  imóveis residenciais plenos (vagas de garagem, boxes, frações), ou")
    L.append("  subdeclaração que escapou da limpeza. O modelo, treinado na massa de")
    L.append("  imóveis normais, projeta um preço 'de imóvel inteiro' e por isso")
    L.append("  superavalia — gerando o erro alto desta faixa.")
    L.append("=" * 70)

    texto = "\n".join(L)
    caminho = OUT / 'atipicos_relatorio.txt'
    caminho.write_text(texto, encoding='utf-8')
    print(f"  salvo: {caminho}")
    print("\n" + texto)


def _fig_dispersao(teste, atip):
    """Real × previsto: atípicos destacados sobre a massa normal."""
    normal = teste[teste['preco_real'] >= LIMITE_ATIPICO]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(normal['preco_real'], normal['preco_previsto'], s=6,
               color=COR['normal'], alpha=0.25, label='demais imóveis', zorder=1)
    ax.scatter(atip['preco_real'], atip['preco_previsto'], s=70,
               color=COR['atipico'], edgecolor='white', linewidth=0.6,
               label=f'atípicos (< R$ {LIMITE_ATIPICO//1000}k)  n={len(atip)}',
               zorder=3)
    lim = max(teste['preco_real'].quantile(0.99), atip['preco_previsto'].max())
    ax.plot([0, lim], [0, lim], ls='--', lw=1.2, color='#444441',
            label='previsão perfeita', zorder=2)
    ax.axvline(LIMITE_ATIPICO, ls=':', lw=1, color=COR['atipico'], alpha=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(FMT_MILHAR); ax.yaxis.set_major_formatter(FMT_MILHAR)
    ax.set_xlabel("preço real (R$)"); ax.set_ylabel("preço previsto (R$)")
    ax.set_title("Imóveis atípicos: o modelo superavalia os de baixo valor")
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    caminho = OUT / 'atipicos_dispersao.png'
    fig.savefig(caminho, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  salvo: {caminho}")


def _fig_contexto(teste, atip):
    """Área e preço/m²: onde os atípicos caem na distribuição geral."""
    tem_area = 'area_construida_m2' in teste.columns
    tem_pm2 = 'preco_m2_real' in teste.columns
    n_pain = tem_area + tem_pm2
    if n_pain == 0:
        return
    fig, axes = plt.subplots(1, n_pain, figsize=(5.5 * n_pain, 4.5))
    if n_pain == 1:
        axes = [axes]
    i = 0
    if tem_area:
        ax = axes[i]; i += 1
        ax.hist(teste['area_construida_m2'].clip(upper=400), bins=50,
                color=COR['normal'], alpha=0.6, density=True, label='geral')
        for v in atip['area_construida_m2']:
            ax.axvline(min(v, 400), color=COR['atipico'], alpha=0.6, lw=0.8)
        ax.set_xlabel("área construída (m²)"); ax.set_ylabel("densidade")
        ax.set_title("Área: atípicos (linhas) vs. geral")
        ax.legend()
    if tem_pm2:
        ax = axes[i]
        ax.hist(teste['preco_m2_real'].clip(upper=15000), bins=50,
                color=COR['normal'], alpha=0.6, density=True, label='geral')
        for v in atip['preco_m2_real']:
            ax.axvline(min(v, 15000), color=COR['atipico'], alpha=0.6, lw=0.8)
        ax.xaxis.set_major_formatter(FMT_MILHAR)
        ax.set_xlabel("preço/m² real (R$)"); ax.set_ylabel("densidade")
        ax.set_title("Preço/m²: atípicos (linhas) vs. geral")
        ax.legend()
    caminho = OUT / 'atipicos_contexto.png'
    fig.savefig(caminho, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  salvo: {caminho}")


def main():
    print(f"Carregando dataset final: {ITBI_FINAL}")
    df = pd.read_csv(ITBI_FINAL)
    teste = _preparar(df)
    atip = _tabela_atipicos(teste)
    if len(atip) == 0:
        print(f"Nenhum imóvel com preço real < R$ {LIMITE_ATIPICO:,.0f} no holdout "
              f"(>= {ANO_TESTE_INICIO}). Nada a relatar.")
        return
    print(f"\n[1/4] Tabela completa dos atípicos...")
    _salvar_tabela(atip)
    print(f"\n[2/4] Relatório diagnóstico...")
    _relatorio_texto(teste, atip)
    print(f"\n[3/4] Figura de dispersão (real × previsto)...")
    _fig_dispersao(teste, atip)
    print(f"\n[4/4] Figura de contexto (área, preço/m²)...")
    _fig_contexto(teste, atip)
    print(f"\nPronto. Tudo em {OUT}")


if __name__ == '__main__':
    main()