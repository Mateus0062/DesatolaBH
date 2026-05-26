from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.modeling.train import pipeline_completo
from src.modeling.evaluate import avaliar_modelos_teste

print("Executando pipeline completo...")
print("(Isso vai treinar os modelos novamente)\n")

# Treinar e obter dados de teste
modelos, resultados, (X_test, y_test) = pipeline_completo()

print("\n" + "="*80)
print("AGORA AVALIANDO NO CONJUNTO DE TESTE...")
print("="*80)

# Avaliar no conjunto de teste
resultados_teste = avaliar_modelos_teste(X_test, y_test)