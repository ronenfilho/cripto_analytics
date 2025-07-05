from src.models import walk_forward_prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
import subprocess


def test_walk_forward_prediction():
    # Cria dados de exemplo
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])
    model = LinearRegression()

    # Executa a função
    predictions = walk_forward_prediction(model, X, y, min_train_size=3)

    # Verifica o resultado
    assert len(predictions) == 2  # Deve haver 2 previsões
      
