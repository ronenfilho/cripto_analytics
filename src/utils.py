import time
import functools

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Iniciando processamento: {func.__name__}...")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Processamento concluído em {elapsed:.2f} segundos.")
            return result
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
            raise
        finally:
            print(f"Finalizando execução: {func.__name__}")
    return wrapper