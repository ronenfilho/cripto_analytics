from src.main import main
import types
from unittest.mock import patch, MagicMock
import subprocess


def test_main_pipeline_all_steps():
    args = types.SimpleNamespace(
        data=False,
        features=False,
        model=False,
        simulate=False,
        crypto=None,
        days=None,
        capital=None
    )

    with patch("main.update_env_variable"), \
         patch("config.TEST_PERIOD_DAYS", 30), \
         patch("config.SYMBOLS", ["BTC/USDT"]), \
         patch("main.logging.info"), \
         patch("main.logging.critical"), \
         patch("main.__name__", "__main__"), \
         patch("sys.argv", ["main.py"]), \
         patch("src.data_load.main") as mock_data, \
         patch("src.features.main") as mock_feat, \
         patch("src.models.main", return_value=(MagicMock(), MagicMock())) as mock_model, \
         patch("src.simulate.main") as mock_sim:

        # Simula a execução do pipeline
        from src.main import main
        main()

        # Verifica se as etapas foram chamadas
        mock_data.assert_called_once()
        mock_feat.assert_called_once()
        mock_model.assert_called_once()
        mock_sim.assert_called_once()


def test_run_as_script():
    result = subprocess.run(["python", "-m", "src.main"], capture_output=True, text=True)
    assert result.returncode == 0
