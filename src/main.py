import os
import sys  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.data_load as data_load
import src.features as features
import src.models as models
import src.simulate as simulate

if __name__ == "__main__":
    print("Executando o pipeline de análise de criptomoedas...")

    # Carregar dados
    data_load.main()

    # Calcular features
    features.main()

    # Treinar modelos
    #models, data_calculate = models.main()

    # Simular investimentos
    simulate.main()

