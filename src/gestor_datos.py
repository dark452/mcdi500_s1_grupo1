import pandas as pd
class GestorDatos:
    """Clase encargada exclusivamente de la carga y exportación de archivos."""
    def __init__(self, ruta_entrada: str, ruta_salida: str):
        self.ruta_entrada = ruta_entrada
        self.ruta_salida = ruta_salida

    def cargar_datos(self) -> pd.DataFrame:
        """Carga dataset raw, desde un archivo CSV

        Parámetros
        ----------
        file_path : str
            Ruta del archivo CSV utilizado como entrada.

        Retorno
        -------
        pd.DataFrame
            Datos cargados en un DataFrame.

        Excepción
        ---------
        FileNotFoundError
            Si la ruta al archivo CSV no existe. Se muestra un mensaje de error
        """
        try:
            df = pd.read_csv(self.ruta_entrada)
            print(f"[OK] Datos cargados exitosamente desde {self.ruta_entrada}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
            f"[ERROR] No se encontró el archivo en {self.ruta_entrada}"
            "Verificar que el archivo CSV se encuentre en data/raw."
        )

    def guardar_datos(self, df: pd.DataFrame) -> None:
        """Exporta el dataset procesado en formato CSV en la ruta recibida como parámetro

        Parámetros
        ----------
        df        : pd.DataFrame procesado
        file_path : Ruta de salida del archivo CSV
        """
         
        df.to_csv(self.ruta_salida, index=False)
        print(f"[OK] Dataset procesado exportado a: {self.ruta_salida}")
        print(f'     Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas')
        print(f'     Memoria: {df.memory_usage(deep=True).sum() / 1024:.1f} KB')