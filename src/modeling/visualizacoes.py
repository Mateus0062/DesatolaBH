import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import OUTPUTS_MODELS, OUTPUTS_FIGURES, OUTPUTS_TABLES

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def carregar_modelo_e_dados_teste():
    print("Carregando modelo Random Forest e dados de teste...")

    # Carregar modelo
    modelo_path = OUTPUTS_MODELS / 'random_forest.pkl'
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)

    print(f"  ✓ Modelo carregado de {modelo_path}")

    # Carregar dados de teste (precisa rodar train.py primeiro e salvar)
    # Por enquanto, vamos usar uma abordagem alternativa
    print("\n⚠️  Para gerar gráficos, execute:")
    print("    python gerar_graficos.py")

    return modelo


def criar_scatter_plot(y_true, y_pred, modelo_nome='Random Forest'):
    print(f"\n[1/4] Criando scatter plot ({modelo_nome})...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='#3498db', edgecolors='none')

    # Linha diagonal (predição perfeita)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Predição perfeita', alpha=0.8)

    # Formatação
    ax.set_xlabel('Valor Real (R$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Previsto (R$)', fontsize=12, fontweight='bold')
    ax.set_title(f'Predição vs Realidade — {modelo_nome}',
                 fontsize=14, fontweight='bold', pad=15)

    # Formatar eixos para milhares
    ax.ticklabel_format(style='plain', axis='both')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.0f}k'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.0f}k'))

    # Calcular métricas para adicionar no gráfico
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Adicionar caixa com métricas
    textstr = f'R² = {r2:.4f}\nMAE = R$ {mae:,.0f}\nMAPE = {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salvar
    filename = f'scatter_plot_{modelo_nome.lower().replace(" ", "_")}.png'
    filepath = OUTPUTS_FIGURES / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  ✓ Salvo: {filename}")

    plt.close()


def criar_residual_plot(y_true, y_pred, modelo_nome='Random Forest'):
    print(f"\n[2/4] Criando residual plot ({modelo_nome})...")

    # Calcular resíduos
    residuos = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter dos resíduos
    ax.scatter(y_pred, residuos, alpha=0.3, s=10, color='#e74c3c', edgecolors='none')

    # Linha horizontal em zero
    ax.axhline(y=0, color='black', linestyle='--', lw=2, alpha=0.8, label='Resíduo = 0')

    # Formatação
    ax.set_xlabel('Valor Previsto (R$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resíduo (Real - Previsto)', fontsize=12, fontweight='bold')
    ax.set_title(f'Análise de Resíduos — {modelo_nome}',
                 fontsize=14, fontweight='bold', pad=15)

    # Formatar eixos
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.0f}k'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.0f}k'))

    # Adicionar informação sobre distribuição
    media_residuo = residuos.mean()
    std_residuo = residuos.std()

    textstr = f'Média = R$ {media_residuo:,.0f}\nStd = R$ {std_residuo:,.0f}'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salvar
    filename = f'residual_plot_{modelo_nome.lower().replace(" ", "_")}.png'
    filepath = OUTPUTS_FIGURES / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  ✓ Salvo: {filename}")

    plt.close()


def criar_distribuicao_erros(y_true, y_pred, modelo_nome='Random Forest'):
    print(f"\n[3/4] Criando distribuição de erros ({modelo_nome})...")

    # Calcular erros percentuais
    erros_percent = np.abs((y_true - y_pred) / y_true) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histograma
    n, bins, patches = ax.hist(erros_percent, bins=50, edgecolor='black',
                               alpha=0.7, color='#2ecc71')

    # Linha vertical na mediana
    mediana = erros_percent.median()
    ax.axvline(x=mediana, color='red', linestyle='--', lw=2,
               label=f'Mediana: {mediana:.2f}%', alpha=0.8)

    # Formatação
    ax.set_xlabel('Erro Percentual (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribuição de Erros — {modelo_nome}',
                 fontsize=14, fontweight='bold', pad=15)

    # Calcular estatísticas
    pct_5 = (erros_percent < 5).sum() / len(erros_percent) * 100
    pct_10 = (erros_percent < 10).sum() / len(erros_percent) * 100
    pct_15 = (erros_percent < 15).sum() / len(erros_percent) * 100

    textstr = (f'Erro < 5%:  {pct_5:.1f}%\n'
               f'Erro < 10%: {pct_10:.1f}%\n'
               f'Erro < 15%: {pct_15:.1f}%')
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Salvar
    filename = f'distribuicao_erros_{modelo_nome.lower().replace(" ", "_")}.png'
    filepath = OUTPUTS_FIGURES / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  ✓ Salvo: {filename}")

    plt.close()


def criar_comparacao_modelos():
    """
    Gráfico 4: Comparação de modelos (bar chart)
    """
    print(f"\n[4/4] Criando comparação de modelos...")

    # Carregar resultados
    df_resultados = pd.read_csv(OUTPUTS_TABLES / 'resultados_teste.csv')

    # Preparar dados
    modelos = df_resultados['Modelo'].tolist()
    mae = df_resultados['MAE'].tolist()
    rmse = df_resultados['RMSE'].tolist()
    mape = df_resultados['MAPE'].tolist()

    # Criar subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cores = ['#3498db', '#e74c3c', '#f39c12']

    # MAE
    axes[0].bar(modelos, mae, color=cores, alpha=0.8, edgecolor='black')
    axes[0].set_title('MAE (Erro Médio Absoluto)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE (R$)', fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.1f}k'))

    # RMSE
    axes[1].bar(modelos, rmse, color=cores, alpha=0.8, edgecolor='black')
    axes[1].set_title('RMSE (Raiz do Erro Quadrático)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE (R$)', fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x / 1000:.1f}k'))

    # MAPE
    axes[2].bar(modelos, mape, color=cores, alpha=0.8, edgecolor='black')
    axes[2].set_title('MAPE (Erro Percentual Médio)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Comparação de Performance dos Modelos',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Salvar
    filename = 'comparacao_modelos.png'
    filepath = OUTPUTS_FIGURES / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  ✓ Salvo: {filename}")

    plt.close()


def gerar_todos_graficos(y_true, y_pred, modelo_nome='Random Forest'):
    print("=" * 80)
    print("GERANDO GRÁFICOS PARA O ARTIGO")
    print("=" * 80)

    criar_scatter_plot(y_true, y_pred, modelo_nome)
    criar_residual_plot(y_true, y_pred, modelo_nome)
    criar_distribuicao_erros(y_true, y_pred, modelo_nome)
    criar_comparacao_modelos()

    print("\n" + "=" * 80)
    print("GRÁFICOS GERADOS COM SUCESSO!")
    print("=" * 80)
    print(f"\nArquivos salvos em: {OUTPUTS_FIGURES}")
    print("\nGráficos criados:")
    print("  1. scatter_plot_random_forest.png")
    print("  2. residual_plot_random_forest.png")
    print("  3. distribuicao_erros_random_forest.png")
    print("  4. comparacao_modelos.png")


if __name__ == '__main__':
    print("Execute via script gerar_graficos.py")