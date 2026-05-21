import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import OUTPUTS_MODELS

class RecomendadorImoveis:
    def __init__(self, modelo_path=None):
        """Carrega modelo treinado"""
        if modelo_path is None:
            modelo_path = OUTPUTS_MODELS / 'random_forest.pkl'

        with open(modelo_path, 'rb') as f:
            self.modelo = pickle.load(f)

        print(f"✓ Modelo carregado de {modelo_path}")

    def prever_preco_justo(self, features):
        return self.modelo.predict(features)[0]

    def calcular_desvio(self, preco_pedido, preco_justo):
        return ((preco_pedido - preco_justo) / preco_justo) * 100

    def recomendar_acao(self, preco_pedido, preco_justo, tempo_parado_dias,
                        valorizacao_bairro, idade_imovel):
        desvio = self.calcular_desvio(preco_pedido, preco_justo)

        if desvio > 15:
            if tempo_parado_dias > 180:
                return {
                    'acao': 'REDUZIR PREÇO URGENTE',
                    'emoji': '🔴',
                    'preco_sugerido': preco_justo * 1.05,  # 5% acima do justo
                    'reducao_necessaria': preco_pedido - (preco_justo * 1.05),
                    'justificativa': f"Imóvel {desvio:.1f}% acima do mercado e parado há {tempo_parado_dias} dias. "
                                     f"Reduzir para próximo do preço justo aumenta chances de venda.",
                    'prioridade': 'ALTA'
                }
            else:
                return {
                    'acao': 'REDUZIR PREÇO',
                    'emoji': '🟠',
                    'preco_sugerido': preco_justo * 1.10,  # 10% acima do justo
                    'reducao_necessaria': preco_pedido - (preco_justo * 1.10),
                    'justificativa': f"Imóvel {desvio:.1f}% acima do mercado. "
                                     f"Ajuste de preço recomendado para acelerar venda.",
                    'prioridade': 'MÉDIA'
                }

        # ─── REGRA 2: Preço levemente acima (5-15%) ───────────────────────
        elif 5 < desvio <= 15:
            if valorizacao_bairro > 10:
                return {
                    'acao': 'AGUARDAR',
                    'emoji': '🟡',
                    'preco_sugerido': preco_pedido,
                    'reducao_necessaria': 0,
                    'justificativa': f"Bairro em valorização ({valorizacao_bairro:.1f}% nos últimos 3 anos). "
                                     f"Preço {desvio:.1f}% acima do mercado pode se ajustar naturalmente.",
                    'prioridade': 'BAIXA'
                }
            else:
                return {
                    'acao': 'REDUZIR LEVEMENTE',
                    'emoji': '🟠',
                    'preco_sugerido': preco_justo * 1.05,
                    'reducao_necessaria': preco_pedido - (preco_justo * 1.05),
                    'justificativa': f"Preço {desvio:.1f}% acima do mercado. "
                                     f"Pequeno ajuste pode aumentar competitividade.",
                    'prioridade': 'MÉDIA'
                }

        # ─── REGRA 3: Preço no ponto (±5%) ────────────────────────────────
        elif -5 <= desvio <= 5:
            if tempo_parado_dias > 365:
                return {
                    'acao': 'CONSIDERAR ALUGUEL',
                    'emoji': '🔵',
                    'preco_sugerido': None,
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço está justo (desvio {desvio:.1f}%), mas parado há {tempo_parado_dias} dias. "
                                     f"Considerar alugar pode ser mais viável que vender neste momento.",
                    'prioridade': 'MÉDIA'
                }
            else:
                return {
                    'acao': 'MANTER PREÇO',
                    'emoji': '🟢',
                    'preco_sugerido': preco_pedido,
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço está alinhado com o mercado (desvio {desvio:.1f}%). "
                                     f"Manter estratégia atual e aguardar comprador.",
                    'prioridade': 'BAIXA'
                }

        # ─── REGRA 4: Preço abaixo do mercado (<-5%) ──────────────────────
        else:  # desvio < -5
            if idade_imovel > 30:
                return {
                    'acao': 'REFORMAR ANTES DE VENDER',
                    'emoji': '🔨',
                    'preco_sugerido': preco_justo * 0.90,  # vender 10% abaixo após reforma
                    'reducao_necessaria': 0,
                    'justificativa': f"Imóvel antigo ({idade_imovel} anos) com preço {abs(desvio):.1f}% abaixo do mercado. "
                                     f"Reforma pode aumentar valor e acelerar venda.",
                    'prioridade': 'MÉDIA'
                }
            else:
                return {
                    'acao': 'AUMENTAR PREÇO',
                    'emoji': '⬆️',
                    'preco_sugerido': preco_justo * 0.95,  # 5% abaixo do justo (margem segura)
                    'reducao_necessaria': 0,
                    'justificativa': f"Preço {abs(desvio):.1f}% abaixo do mercado. "
                                     f"Há margem para aumentar sem prejudicar venda.",
                    'prioridade': 'BAIXA'
                }

    def analisar_imovel(self, dados_imovel, preco_pedido, tempo_parado_dias=0):
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
            tempo_parado_dias=tempo_parado_dias,
            valorizacao_bairro=valorizacao_bairro,
            idade_imovel=idade_imovel
        )

        # Compilar análise completa
        analise = {
            'preco_pedido': preco_pedido,
            'preco_justo_previsto': preco_justo,
            'desvio_percentual': self.calcular_desvio(preco_pedido, preco_justo),
            'tempo_parado_dias': tempo_parado_dias,
            **recomendacao
        }

        return analise

    def gerar_relatorio(self, analise):
        print("\n" + "=" * 80)
        print("RELATÓRIO DE ANÁLISE DO IMÓVEL")
        print("=" * 80)

        print(f"\n📊 AVALIAÇÃO:")
        print(f"  Preço pedido:         R$ {analise['preco_pedido']:>12,.2f}")
        print(f"  Preço justo (IA):     R$ {analise['preco_justo_previsto']:>12,.2f}")
        print(f"  Desvio:               {analise['desvio_percentual']:>12.1f}%")
        print(f"  Tempo no mercado:     {analise['tempo_parado_dias']:>12} dias")

        print(f"\n{analise['emoji']} RECOMENDAÇÃO: {analise['acao']}")
        print(f"  Prioridade: {analise['prioridade']}")

        if analise['preco_sugerido'] is not None:
            print(f"\n💰 PREÇO SUGERIDO: R$ {analise['preco_sugerido']:,.2f}")
            if analise['reducao_necessaria'] > 0:
                print(f"  Redução necessária: R$ {analise['reducao_necessaria']:,.2f} "
                      f"({analise['reducao_necessaria'] / analise['preco_pedido'] * 100:.1f}%)")

        print(f"\n📝 JUSTIFICATIVA:")
        print(f"  {analise['justificativa']}")

        print("\n" + "=" * 80)


# ─── EXEMPLO DE USO ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Sistema Recomendador")