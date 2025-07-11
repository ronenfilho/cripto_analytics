from src.features import calculate_summary_statistics
import pandas as pd
import subprocess
import tempfile
import os


def test_calculate_summary_statistics():
    # Cria um DataFrame de exemplo
    data = pd.DataFrame(
        {"symbol": ["BTC", "BTC", "ETH", "ETH"], "close": [100, 200, 300, 400]}
    )

    # Cria um arquivo temporário para salvar o resultado
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "test_summary.csv")
        # Executa a função com o arquivo temporário
        result = calculate_summary_statistics(data, temp_file)

    # Verifica o resultado
    assert "mean" in result.columns
    assert len(result) == 2  # Deve haver 2 símbolos únicos

def test_run_as_script():
    result = subprocess.run(["python", "-m", "src.features"], capture_output=True, text=True)
    assert result.returncode == 0    
