import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import OUTPUTS_MODELS

class RecomendadorImoveis:
    def __init__(self, modelo_path=None):
        if modelo_path is None:
            modelo_path = OUTPUTS_MODELS / 'xgboost.pkl'

        with open(modelo_path, 'rb') as f:
            self.modelo = pickle.load(f)

        print(f"Modelo carregado de {modelo_path}")

    def prever_preco_justo(self, features):
        X = features[self.modelo.feature_names_in_]
        X = X.apply(pd.to_numeric, errors='coerce')
        return np.expm1(self.modelo.predict(X)[0])

    def calcular_desvio(self, preco_pedido, preco_justo):
        return ((preco_pedido - preco_justo) / preco_justo) * 100

    def recomendar_acao(self, preco_pedido, preco_justo, valorizacao_bairro, idade_imovel):

        desvio = self.calcular_desvio(preco_pedido, preco_justo)

        # ─── REGRA 1: Preço muito acima do mercado (> 15%) ───────────────
        if desvio > 15:
            return {
                'acao': 'REDUZIR PREÇO',
                'preco_sugerido': preco_justo * 1.05,
                'reducao_necessaria': preco_pedido - (preco_justo * 1.05),
                'justificativa': f"Preço {desvio:.1f}% acima do valor estimado de "
                                 f"mercado. Ajuste recomendado para tornar o "
                                 f"imóvel competitivo.",
                'prioridade': 'ALTA'
            }

        # ─── REGRA 2: Preço levemente acima (5% a 15%) ───────────────────
        elif 5 < desvio <= 15:
            if valorizacao_bairro > 10:
                return {
                    'acao': 'AGUARDAR',
                    'preco_sugerido': preco_pedido,
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço {desvio:.1f}% acima do mercado, mas o "
                                     f"bairro valorizou {valorizacao_bairro:.1f}% nos "
                                     f"últimos anos. A diferença pode se ajustar "
                                     f"naturalmente.",
                    'prioridade': 'BAIXA'
                }
            else:
                return {
                    'acao': 'REDUZIR LEVEMENTE',
                    'preco_sugerido': preco_justo * 1.05,
                    'reducao_necessaria': preco_pedido - (preco_justo * 1.05),
                    'justificativa': f"Preço {desvio:.1f}% acima do mercado. Pequeno "
                                     f"ajuste aumenta a competitividade.",
                    'prioridade': 'MÉDIA'
                }

        # ─── REGRA 3: Preço alinhado com o mercado (±5%) ─────────────────
        elif -5 <= desvio <= 5:
            return {
                'acao': 'MANTER PREÇO',
                'preco_sugerido': preco_pedido,
                'reducao_necessaria': 0,
                'justificativa': f"Preço alinhado com o valor estimado de mercado "
                                 f"(desvio {desvio:.1f}%). Estratégia adequada.",
                'prioridade': 'BAIXA'
            }

        # ─── REGRA 4: Preço abaixo do mercado (< -5%) ────────────────────
        else:
            if idade_imovel > 30:
                return {
                    'acao': 'REFORMAR ANTES DE VENDER',
                    'preco_sugerido': preco_justo * 0.95,
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço {abs(desvio):.1f}% abaixo do mercado, em "
                                     f"imóvel de {idade_imovel:.0f} anos. Uma reforma "
                                     f"pode elevar o valor de venda.",
                    'prioridade': 'MÉDIA'
                }
            else:
                return {
                    'acao': 'AUMENTAR PREÇO',
                    'preco_sugerido': preco_justo * 0.95,
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço {abs(desvio):.1f}% abaixo do valor "
                                     f"estimado de mercado. Há margem para reajuste.",
                    'prioridade': 'BAIXA'
                }

    def analisar_imovel(self, dados_imovel, preco_pedido):
        # Converter para DataFrame se necessário
        if isinstance(dados_imovel, dict):
            dados_imovel = pd.DataFrame([dados_imovel])

        # Prever preço justo
        preco_justo = self.prever_preco_justo(dados_imovel)

        # Extrair features para regras de decisão
        valorizacao_bairro = dados_imovel['valorizacao_bairro_3anos'].values[0]
        idade_imovel = dados_imovel['idade_imovel'].values[0]

        # Obter recomendação
        recomendacao = self.recomendar_acao(
            preco_pedido=preco_pedido,
            preco_justo=preco_justo,
            valorizacao_bairro=valorizacao_bairro,
            idade_imovel=idade_imovel
        )

        # Compilar análise completa
        analise = {
            'preco_pedido': preco_pedido,
            'preco_justo_previsto': preco_justo,
            'desvio_percentual': self.calcular_desvio(preco_pedido, preco_justo),
            **recomendacao
        }

        return analise

    def gerar_relatorio(self, analise):
        print("\n" + "=" * 80)
        print("RELATÓRIO DE ANÁLISE DO IMÓVEL")
        print("=" * 80)

        print(f"\nAVALIAÇÃO:")
        print(f"Preço pedido: R$ {analise['preco_pedido']:>12,.2f}")
        print(f"Preço justo (IA): R$ {analise['preco_justo_previsto']:>12,.2f}")
        print(f"Desvio: {analise['desvio_percentual']:>12.1f}%")

        print(f"\nRECOMENDAÇÃO: {analise['acao']}")
        print(f"Prioridade: {analise['prioridade']}")

        if analise['preco_sugerido']:
            print(f"\nPREÇO SUGERIDO: R$ {analise['preco_sugerido']:,.2f}")
            if analise['reducao_necessaria'] > 0:
                print(f"Redução necessária: R$ {analise['reducao_necessaria']:,.2f} "
                      f"({analise['reducao_necessaria'] / analise['preco_pedido'] * 100:.1f}%)")

        print(f"\nJUSTIFICATIVA:")
        print(f"{analise['justificativa']}")

        print("\n" + "=" * 80)


# ─── EXEMPLO DE USO ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Sistema Recomendador")