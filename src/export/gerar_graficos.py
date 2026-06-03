import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sem display, salva direto em arquivo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, ITBI_RAW, OUTPUTS_FIGURES, DATA_MUDANCA_REGIME, TARGET, OUTPUTS_MODELS, OUTPUTS_TABLES, OUTPUTS_MODELS_TRAIN2)

DPI = 300

# ── Caminhos do projeto ───────────────────────────────────────────────────────
MODELO_LGBM = OUTPUTS_MODELS_TRAIN2 / 'lightgbm.pkl'  # modelo de operação (train2.py)
RESULTADOS_DIR = OUTPUTS_TABLES / 'resultados_gerar_grafico'

# ── Parâmetros ────────────────────────────────────────────────────────────────
ANO_TESTE_INICIO = 2023
TARGET_EM_LOG = False
MODELO_PREVE_LOG = True
MIN_TRANSACOES_MES = 30

# ── Nomes de coluna no dataset FINAL (snake_case) ─────────────────────────────
COL_DATA = "data_transacao"      # se não houver data completa, uso COL_ANO + COL_MES
COL_ANO = "ano_transacao"
COL_MES = "mes_transacao"        # opcional; se ausente, fig1 cai para granularidade anual
COL_PRECO_M2 = "preco_m2"
COL_LAT = "lat"
COL_LON = "lon"

# ── Nomes de coluna no dataset RAW (schema original do ITBI; difere do final) ──
COL_RAW_BASE = "Valor Base Calculo"
COL_RAW_DECLARADO = "Valor Declarado"
COL_RAW_AREA = "Area Construida Adquirida"   # numerador do preço/m² = declarado / área
COL_RAW_DATA = "Data Quitacao Transacao"

# ── Import dos helpers do projeto (para reusar o MESMO _Xy do treino) ─────────
MODULO_XY = "src.sistema_decisao.recomendador"   # de onde importar _Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ESTILO — paleta sóbria, consistente, pronta para impressão
# ──────────────────────────────────────────────────────────────────────────────
COR = {
    "primaria": "#2C5F8A",    # azul (modelo / destaque)
    "secundaria": "#C45A3B",  # terracota (baseline / contraste)
    "neutra": "#6B6B68",      # cinza
    "verde": "#1D9E75",       # dentro da faixa / positivo
    "vermelho": "#A32D2D",    # caro / alerta
    "ambar": "#B8860B",       # barato / atenção
    "grade": "#D9D9D6",
    "regime": "#A32D2D",      # linha vertical da quebra de regime
}


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.edgecolor": "#444441", "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": COR["grade"], "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 10,
        "xtick.color": "#444441", "ytick.color": "#444441",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


# ─── Formatadores em pt-BR (R$ 350.000) ────────────────────────────────────────
def _fmt_reais(x, _pos=None):
    return f"R$ {x:,.0f}".replace(",", ".")


def _fmt_reais_compacto(x, _pos=None):
    """R$ compacto para eixos com muitos ticks: R$ 250k, R$ 1,5M."""
    if abs(x) >= 1_000_000:
        return f"R$ {x/1_000_000:.1f}M".replace(".", ",")
    if abs(x) >= 1_000:
        return f"R$ {x/1_000:.0f}k"
    return f"R$ {x:.0f}"


def _fmt_milhar(x, _pos=None):
    return f"{x:,.0f}".replace(",", ".")


def _fmt_pct(x, _pos=None):
    return f"{x:.0f}%"


FMT_REAIS = mticker.FuncFormatter(_fmt_reais)
FMT_REAIS_COMPACTO = mticker.FuncFormatter(_fmt_reais_compacto)
FMT_MILHAR = mticker.FuncFormatter(_fmt_milhar)
FMT_PCT = mticker.FuncFormatter(_fmt_pct)


def _salvar(fig, nome):
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    caminho = OUTPUTS_FIGURES / nome
    fig.savefig(caminho)
    plt.close(fig)
    logger.info("salvo: %s", caminho)
    return caminho


def _linha_regime(ax, label=True):
    ax.axvline(DATA_MUDANCA_REGIME, color=COR["regime"], ls="--", lw=1.4, zorder=5)
    if label:
        ax.annotate("mudança de regime\n(jul/2023)",
                    xy=(DATA_MUDANCA_REGIME, ax.get_ylim()[1]),
                    xytext=(6, -6), textcoords="offset points",
                    va="top", ha="left", fontsize=9, color=COR["regime"])


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — QUEBRA DE REGIME
# ══════════════════════════════════════════════════════════════════════════════
def fig1_quebra_regime(mensal_raw, erro_mensal):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    ax1.plot(mensal_raw.index, mensal_raw["razao_base_declarado"],
             color=COR["primaria"], lw=1.8, label="razão base/declarado (mediana)")
    ax1.set_ylabel("razão base / declarado", color=COR["primaria"])
    ax1.tick_params(axis="y", labelcolor=COR["primaria"])
    ax1.axhline(1.0, color=COR["neutra"], ls=":", lw=1)

    ax1b = ax1.twinx()
    ax1b.plot(mensal_raw.index, mensal_raw["pct_base_igual_declarado"],
              color=COR["secundaria"], lw=1.4, alpha=0.85, label="% base = declarado")
    ax1b.yaxis.set_major_formatter(FMT_PCT)
    ax1b.set_ylabel("% base = declarado", color=COR["secundaria"])
    ax1b.tick_params(axis="y", labelcolor=COR["secundaria"])
    ax1b.grid(False)
    ax1.set_title("(a) Evidência da quebra estrutural no alvo")
    _linha_regime(ax1)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center left")

    ax2.plot(erro_mensal.index, erro_mensal["mape"], color=COR["primaria"], lw=1.6)
    ax2.fill_between(erro_mensal.index, erro_mensal["mape"], alpha=0.12, color=COR["primaria"])
    ax2.yaxis.set_major_formatter(FMT_PCT)
    ax2.set_ylabel("MAPE do modelo")
    ax2.set_xlabel("mês")
    ax2.set_title("(b) Erro do modelo ao longo do tempo")
    _linha_regime(ax2, label=False)
    return _salvar(fig, "fig1_quebra_regime.png")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — DADOS / LIMPEZA
# ══════════════════════════════════════════════════════════════════════════════
def fig2a_dist_preco_m2(preco_m2_raw, preco_m2_limpo, clip=(0, 25000)):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(clip[0], clip[1], 60)
    ax.hist(np.clip(preco_m2_raw, *clip), bins=bins, color=COR["neutra"], alpha=0.55,
            label=f"bruto (n={len(preco_m2_raw):,})".replace(",", "."), density=True)
    ax.hist(np.clip(preco_m2_limpo, *clip), bins=bins, color=COR["primaria"], alpha=0.7,
            label=f"limpo (n={len(preco_m2_limpo):,})".replace(",", "."), density=True)
    ax.set_xlabel("preço por m² (R$)")
    ax.set_ylabel("densidade")
    ax.xaxis.set_major_formatter(FMT_MILHAR)
    ax.set_title("Distribuição do preço/m² antes e depois da limpeza")
    ax.legend()
    return _salvar(fig, "fig2a_dist_preco_m2.png")


def fig2b_mapa_calor_bh(lon, lat, preco_m2, gridsize=45, mincnt=5):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    hb = ax.hexbin(lon, lat, C=preco_m2, reduce_C_function=np.median,
                   gridsize=gridsize, mincnt=mincnt, cmap="viridis", linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("preço/m² mediano (R$)")
    cb.formatter = FMT_MILHAR
    cb.update_ticks()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    ax.set_title("Preço/m² mediano por região de Belo Horizonte")
    return _salvar(fig, "fig2b_mapa_calor_bh.png")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — DESEMPENHO
# ══════════════════════════════════════════════════════════════════════════════
def fig3a_backtest_modelos(df_backtest, destaque="LightGBM"):
    df = df_backtest.sort_values("mape_medio", ascending=False).reset_index(drop=True)
    cores = [COR["primaria"] if m == destaque else COR["neutra"] for m in df["modelo"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["modelo"], df["mape_medio"], xerr=df["mape_std"], color=cores, alpha=0.9,
            capsize=4, error_kw={"elinewidth": 1, "ecolor": "#444441"})
    for y, (v, s) in enumerate(zip(df["mape_medio"], df["mape_std"])):
        ax.text(v + s + 0.3, y, f"{v:.1f}%", va="center", fontsize=10)
    ax.xaxis.set_major_formatter(FMT_PCT)
    ax.set_xlabel("MAPE no backtest (regime novo)")
    ax.set_title("Desempenho por modelo — backtest rolling-origin")
    return _salvar(fig, "fig3a_backtest_modelos.png")


def fig3b_erro_por_trimestre(df_trim):
    x = (df_trim["trimestre"].dt.to_timestamp()
         if hasattr(df_trim["trimestre"], "dt") and hasattr(df_trim["trimestre"].dt, "to_timestamp")
         else pd.to_datetime(df_trim["trimestre"]))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, df_trim["baseline"], color=COR["secundaria"], lw=1.8, marker="o", ms=4,
            label="Baseline R$/m²")
    ax.plot(x, df_trim["lightgbm"], color=COR["primaria"], lw=1.8, marker="s", ms=4,
            label="LightGBM")
    ax.yaxis.set_major_formatter(FMT_PCT)
    ax.set_ylabel("MAPE")
    ax.set_xlabel("trimestre")
    ax.set_title("Erro por trimestre: baseline vs. modelo")
    _linha_regime(ax)
    ax.legend()
    return _salvar(fig, "fig3b_erro_por_trimestre.png")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — ERRO / RECOMENDADOR
# ══════════════════════════════════════════════════════════════════════════════
def fig5a_erro_por_faixa(df_faixa):
    x = np.arange(len(df_faixa))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, df_faixa["mape"], w, color=COR["primaria"], label="MAPE (média)")
    ax.bar(x + w / 2, df_faixa["mdape"], w, color=COR["verde"], label="MdAPE (mediana)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_faixa["faixa"], rotation=15, ha="right")
    ax.yaxis.set_major_formatter(FMT_PCT)
    ax.set_ylabel("erro percentual")
    ax.set_title("Erro do modelo por faixa de preço")
    for i, n in enumerate(df_faixa["n"]):
        ax.text(i, max(df_faixa["mape"].iloc[i], df_faixa["mdape"].iloc[i]) + 1,
                f"n={n:,}".replace(",", "."), ha="center", fontsize=8, color=COR["neutra"])
    ax.legend()
    return _salvar(fig, "fig5a_erro_por_faixa.png")


def fig5b_banda_incerteza(df_ex):
    mapa_cor = {"dentro": COR["verde"], "caro": COR["vermelho"], "barato": COR["ambar"]}
    mapa_lbl = {"dentro": "dentro da faixa", "caro": "acima da faixa (caro)",
                "barato": "abaixo da faixa (barato)"}
    y = np.arange(len(df_ex))
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(df_ex) + 2.2))
    for i, row in df_ex.reset_index(drop=True).iterrows():
        ax.plot([row["p10"], row["p90"]], [i, i], color=COR["grade"], lw=9,
                solid_capstyle="round", zorder=1)
        ax.plot(row["p50"], i, "o", color=COR["primaria"], ms=8, zorder=3)
        ax.plot(row["pedido"], i, "X", color=mapa_cor[row["situacao"]], ms=12,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df_ex["rotulo"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FMT_REAIS_COMPACTO)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_xlabel("preço (R$)")
    ax.set_title("Recomendador: preço pedido vs. faixa de preço justo (p10–p90)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COR["grade"], lw=9, label="faixa p10–p90 (preço justo)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COR["primaria"], ms=9, label="previsão central (p50)"),
    ] + [Line2D([0], [0], marker="X", color="w", markerfacecolor=c, ms=11, label=mapa_lbl[k])
         for k, c in mapa_cor.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, columnspacing=1.4, handletextpad=0.5)
    return _salvar(fig, "fig5b_banda_incerteza.png")


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO REAL — liga ao seu projeto (config.py + modelo)
# ══════════════════════════════════════════════════════════════════════════════
import importlib
import pickle


def _coluna_mes(df, col_data=COL_DATA):
    """Series datetime no 1º dia do mês. col_data permite usar o nome do raw."""
    if col_data in df.columns:
        return pd.to_datetime(df[col_data], errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()
    if COL_ANO in df.columns and COL_MES in df.columns:
        return pd.to_datetime(dict(year=df[COL_ANO].astype(int), month=df[COL_MES].astype(int), day=1))
    if COL_ANO in df.columns:
        logger.warning("Sem mês: fig1 usará granularidade ANUAL (menos nítida).")
        return pd.to_datetime(df[COL_ANO].astype(int).astype(str) + "-01-01")
    raise KeyError(f"Sem coluna de data ('{col_data}') nem COL_ANO+COL_MES.")


def _pred_reais(modelo, X):
    p = np.asarray(modelo.predict(X), dtype=float)
    return np.expm1(p) if MODELO_PREVE_LOG else p


def _alvo_reais(y):
    y = np.asarray(y, dtype=float)
    return np.expm1(y) if TARGET_EM_LOG else y


def _xy(df):
    """Reusa o _Xy do projeto (mesma exclusão/categóricas do treino). Cai para
    uma versão local se o import falhar — ajuste MODULO_XY se isso acontecer."""
    try:
        mod = importlib.import_module(MODULO_XY)
        return mod._Xy(df)
    except Exception as e:
        logger.warning("Não importei _Xy de %s (%s). Usando fallback local — "
                       "confira se bate com o treino.", MODULO_XY, e)
        from config import COLUNAS_EXCLUIR_MODELO, TARGET, FEATURES_CATEGORICAS
        y = df[TARGET].copy()
        aux = ["_data", "_periodo"]
        X = df.drop(columns=[c for c in list(COLUNAS_EXCLUIR_MODELO) + [TARGET] + aux
                             if c in df.columns])
        for c in FEATURES_CATEGORICAS:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X, y


def _carregar_modelo():
    with open(MODELO_LGBM, "rb") as f:
        return pickle.load(f)


def _ler_csv(nome, colunas_esperadas, dica):
    """Lê um CSV canônico. Se faltar (ou faltar coluna), avisa e devolve None —
    a figura correspondente é pulada, sem derrubar o resto."""
    caminho = RESULTADOS_DIR / nome
    if not caminho.exists():
        logger.warning("Pulando figura: faltou %s. %s", caminho, dica)
        return None
    df = pd.read_csv(caminho)
    faltando = set(colunas_esperadas) - set(df.columns)
    if faltando:
        logger.warning("Pulando figura: %s sem as colunas %s.", caminho, faltando)
        return None
    return df


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _num_br(s):
    """Converte para número aceitando formato brasileiro ('1.234,56').
    Colunas já numéricas (caso do final) passam direto, sem mexer no separador."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ─── blocos individuais ─────────────────────────────────────────────────────
def _build_mensal_raw():
    raw = pd.read_csv(ITBI_RAW)
    base = _num_br(raw[COL_RAW_BASE])
    decl = _num_br(raw[COL_RAW_DECLARADO])
    raw = raw.assign(_base=base, _decl=decl)
    raw = raw[(raw["_base"] > 0) & (raw["_decl"] > 0)].copy()
    raw["_mes"] = _coluna_mes(raw, COL_RAW_DATA)
    raw["_razao"] = raw["_base"] / raw["_decl"]
    raw["_igual"] = np.isclose(raw["_base"], raw["_decl"], rtol=1e-3)
    g = raw.dropna(subset=["_mes"]).groupby("_mes")
    out = pd.DataFrame({
        "razao_base_declarado": g["_razao"].median(),
        "pct_base_igual_declarado": g["_igual"].mean() * 100,
        "_n": g.size(),
    }).sort_index()
    # descarta meses esparsos (medianas instáveis com poucas transações)
    out = out[out["_n"] >= MIN_TRANSACOES_MES]
    return out.drop(columns="_n")


def _build_erros_teste(df_final, modelo):
    teste = df_final[df_final[COL_ANO] >= ANO_TESTE_INICIO].copy()
    X, y = _xy(teste)
    pred = _pred_reais(modelo, X)
    real = _alvo_reais(y)
    ape = np.abs(pred - real) / np.maximum(real, 1.0) * 100
    teste = teste.assign(_real=real, _pred=pred, _ape=ape)
    teste["_mes"] = _coluna_mes(teste)
    return teste


def _build_erro_mensal(teste):
    """BLOCO 1b. Preferência: predicoes_backtest.csv (real/pred em R$) cobrindo
    toda a série. Senão, erro do holdout (>= ANO_TESTE_INICIO)."""
    caminho = RESULTADOS_DIR / "predicoes_backtest.csv"
    if caminho.exists():
        bt = pd.read_csv(caminho)
        bt["_mes"] = _coluna_mes(bt)
        bt["_ape"] = np.abs(_num(bt["pred"]) - _num(bt["real"])) / np.maximum(_num(bt["real"]), 1.0) * 100
        return bt.dropna(subset=["_mes"]).groupby("_mes")["_ape"].mean().rename("mape").to_frame()
    logger.warning("Sem predicoes_backtest.csv: fig1(b) mostra só o holdout (>= %d).",
                   ANO_TESTE_INICIO)
    return teste.groupby("_mes")["_ape"].mean().rename("mape").to_frame()


def _serie_preco_m2(df, col_pm2=COL_PRECO_M2, col_valor=None, col_area=None):
    """Preço/m²: usa col_pm2 se existir; senão computa col_valor / col_area.
    Usa parser BR (raw vem com '1.234,56'); colunas numéricas passam direto."""
    if col_pm2 in df.columns:
        return _num_br(df[col_pm2])
    if col_valor and col_area and col_valor in df.columns and col_area in df.columns:
        area = _num_br(df[col_area]).replace(0, np.nan)
        return _num_br(df[col_valor]) / area
    return None


def _build_dist_preco_m2(caminho_raw, df_final):
    raw = pd.read_csv(caminho_raw)
    s_raw = _serie_preco_m2(raw, col_valor=COL_RAW_DECLARADO, col_area=COL_RAW_AREA)
    s_limpo = _serie_preco_m2(df_final)
    if s_limpo is None:
        raise ValueError(f"Sem preço/m² no final. Colunas: {sorted(df_final.columns)}")
    if s_raw is None:
        logger.warning("RAW sem preço/m² (e sem '%s'/'%s' p/ computar). fig2a só com o limpo. "
                       "Colunas do raw: %s", COL_RAW_DECLARADO, COL_RAW_AREA, sorted(raw.columns))
        s_raw = s_limpo
    return s_raw.dropna().values, s_limpo.dropna().values


def _build_geo(df_final):
    s_pm2 = _serie_preco_m2(df_final)
    g = pd.DataFrame({"lon": _num(df_final[COL_LON]), "lat": _num(df_final[COL_LAT]),
                      "pm2": s_pm2}).dropna()
    return g["lon"].values, g["lat"].values, g["pm2"].values


def _build_faixa(teste):
    cortes = [0, 100_000, 250_000, 500_000, 1_000_000, np.inf]
    rotulos = ["< R$ 100k", "100k–250k", "250k–500k", "500k–1M", "> R$ 1M"]
    teste = teste.copy()
    teste["_faixa"] = pd.cut(teste["_real"], bins=cortes, labels=rotulos, right=False)
    g = teste.groupby("_faixa", observed=True)["_ape"]
    out = pd.DataFrame({"mape": g.mean(), "mdape": g.median(), "n": g.size()}) \
        .reindex(rotulos).reset_index().rename(columns={"_faixa": "faixa"})
    return out.dropna(subset=["n"])


def carregar_real():
    logger.info("Carregando dataset final: %s", ITBI_FINAL)
    df_final = pd.read_csv(ITBI_FINAL)
    modelo = _carregar_modelo()
    teste = _build_erros_teste(df_final, modelo)
    logger.info("Holdout: %d imóveis | MAPE=%.2f%% | MdAPE=%.2f%%",
                len(teste), teste["_ape"].mean(), teste["_ape"].median())

    bruto, limpo = _build_dist_preco_m2(ITBI_RAW, df_final)
    lon, lat, preco_m2_geo = _build_geo(df_final)

    # BLOCO 3 / 5b — métricas canônicas (opcionais: pulam se o CSV não existir)
    df_backtest = _ler_csv("backtest_modelos.csv", ["modelo", "mape_medio", "mape_std"],
                           dica="Exporte a tabela comparativa do evaluate.py.")
    df_trim = _ler_csv("erro_por_trimestre.csv", ["trimestre", "baseline", "lightgbm"],
                       dica="Exporte o MAPE por trimestre (baseline e LightGBM).")
    if df_trim is not None:
        df_trim["trimestre"] = pd.to_datetime(df_trim["trimestre"], errors="coerce")
    df_ex = _ler_csv("exemplos_recomendador.csv",
                     ["rotulo", "p10", "p50", "p90", "pedido", "situacao"],
                     dica="Rode o recomendador em alguns imóveis e salve as faixas.")

    return dict(
        mensal_raw=_build_mensal_raw(),
        erro_mensal=_build_erro_mensal(teste),
        preco_raw=bruto, preco_limpo=limpo,
        lon=lon, lat=lat, preco_m2_geo=preco_m2_geo,
        df_backtest=df_backtest, df_trim=df_trim,
        df_faixa=_build_faixa(teste), df_ex=df_ex,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def gerar_todas(d):
    fig1_quebra_regime(d["mensal_raw"], d["erro_mensal"])
    fig2a_dist_preco_m2(d["preco_raw"], d["preco_limpo"])
    fig2b_mapa_calor_bh(d["lon"], d["lat"], d["preco_m2_geo"])
    fig5a_erro_por_faixa(d["df_faixa"])
    if d["df_backtest"] is not None:
        fig3a_backtest_modelos(d["df_backtest"])
    if d["df_trim"] is not None:
        fig3b_erro_por_trimestre(d["df_trim"])
    if d["df_ex"] is not None:
        fig5b_banda_incerteza(d["df_ex"])


def main():
    argparse.ArgumentParser(description="Gera as figuras do artigo (DesatolaBH).").parse_args()
    aplicar_estilo()
    d = carregar_real()
    gerar_todas(d)
    logger.info("Figuras em: %s", OUTPUTS_FIGURES.resolve())


if __name__ == "__main__":
    main()