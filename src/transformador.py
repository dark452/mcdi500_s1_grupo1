import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

#########################################
# Contrato común (clase base abstracta) #
#########################################

class Transformador(ABC):
    """
    Contrato común para todas las transformaciones del pipeline.

    Cada subclase implementa :meth:`aplicar` con su lógica específica.
    El polimorfismo permite que :class:`Pipeline` las trate de forma uniforme.
    """

    @abstractmethod
    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la transformación al DataFrame y lo devuelve modificado.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset de entrada.

        Retorna
        -------
        pd.DataFrame
            Dataset transformado.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


################################
# Transformaciones de limpieza #
################################

class EliminadorColumna(Transformador):
    """
    Elimina una columna sin valor predictivo (p. ej. Student_ID).

    Parámetros
    ----------
    col : str
        Nombre de la columna a eliminar.
    """

    def __init__(self, col: str) -> None:
        self._col = col

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col in df.columns:
            df = df.drop(columns=[self._col])
            print(f"[OK] EliminarColumna: '{self._col}' eliminada. Shape: {df.shape}")
        else:
            print(f"[WARN] EliminarColumna: columna '{self._col}' no encontrada.")
        return df


class CastBooleano(Transformador):
    """
    Convierte una columna booleana a int64 (False ==> 0, True ==> 1).

    Parámetros
    ----------
    col : str
        Nombre de la columna booleana.
    """

    def __init__(self, col: str = "Paid_Subscription") -> None:
        self._col = col

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col not in df.columns:
            print(f"[WARN] CastBooleano: columna '{self._col}' no encontrada.")
            return df
        df = df.copy()
        df[self._col] = df[self._col].to_numpy().astype(int)
        print(f"[OK] CastBooleano: '{self._col}' ==> int64. "
              f"Distribución: {df[self._col].value_counts().to_dict()}")
        return df


#############################
# Codificación de variables #
#############################

class CodificadorOrdinal(Transformador):
    """
    Asigna enteros consecutivos según la jerarquía real de una variable ordinal.

    Preserva la relación de orden, aplicar One-Hot Encoding a una ordinal
    destruiría esa relación y generaría columnas redundantes.

    Parámetros
    ----------
    col : str
        Columna a codificar.
    orden : list[str]
        Categorías en orden ascendente, p. ej. ['Low', 'Medium', 'High'].
    """

    def __init__(self, col: str, orden: list) -> None:
        self._col = col
        self._orden = orden

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col not in df.columns:
            print(f"[WARN] CodificadorOrdinal: columna '{self._col}' no encontrada.")
            return df
        df = df.copy()
        mapping = {cat: idx for idx, cat in enumerate(self._orden)}
        df[self._col] = df[self._col].map(mapping)
        nulos = df[self._col].isnull().sum()
        if nulos > 0:
            print(f"[ERROR] CodificadorOrdinal: '{self._col}' — {nulos} valores sin mapeo.")
        else:
            print(f"[OK] CodificadorOrdinal: '{self._col}' ==> {mapping}")
        return df


class CodificadorOneHot(Transformador):
    """
    Codifica una variable nominal con LabelEncoder + OneHotEncoder.

    Las variables nominales no tienen orden, asignarles un entero directo
    induciría una jerarquía falsa en los modelos.

    Parámetros
    ----------
    col : str
        Columna nominal a codificar.
    nombres : list[str]
        Nombres de las columnas binarias resultantes, en el mismo orden que
        `categorias` (por convención del equipo, orden alfabético).
    categorias : list[str], optional
        Catálogo completo de categorías esperadas para `col`, en el mismo
        orden que `nombres`. Si se omite, se infieren del DataFrame recibido
        en `aplicar()`,  válido únicamente si ese DataFrame contiene garantizadamente 
        todas las categorías posibles, como el dataset completo. 
        Para lotes parciales, pasar `categorias` explícitamente.
    """

    def __init__(self, col: str, nombres: list, categorias: list = None) -> None:
        self._col = col
        self._nombres = nombres
        self._categorias = categorias

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col not in df.columns:
            print(f"[WARN] CodificadorOneHot: columna '{self._col}' no encontrada.")
            return df
        df = df.copy()

        # Catálogo fijo (recomendado) o inferido del lote (legado).
        if self._categorias is not None:
            categorias_fit = sorted(self._categorias)
        else:
            categorias_fit = sorted(df[self._col].unique())
            print(f"[WARN] CodificadorOneHot: '{self._col}' sin catálogo explícito; "
                  f"se infiere del lote recibido ({len(categorias_fit)} categorías). "
                  f"Esto puede fallar con lotes parciales.")

        if len(self._nombres) != len(categorias_fit):
            raise ValueError(
                f"Se esperaban {len(categorias_fit)} nombres para '{self._col}' "
                f"({categorias_fit}), pero se recibieron {len(self._nombres)}."
            )

        # Si el lote contiene una categoría fuera del catálogo, es un error de datos,
        # no un problema de tamaño de lote: se reporta explícitamente.
        categorias_en_lote = set(df[self._col].unique())
        desconocidas = categorias_en_lote - set(categorias_fit)
        if desconocidas:
            raise ValueError(
                f"'{self._col}' contiene categorías fuera del catálogo esperado: "
                f"{sorted(desconocidas)}. Catálogo: {categorias_fit}."
            )

        le = LabelEncoder()
        le.fit(categorias_fit)
        datos_le = le.transform(df[self._col]).reshape(-1, 1)

        # categories= fuerza a OneHotEncoder a generar una columna por cada
        # categoría del catálogo, incluso si no aparece en este lote.
        ohe = OneHotEncoder(categories=[np.arange(len(categorias_fit))], sparse_output=False)
        matriz = ohe.fit_transform(datos_le)

        nuevas = pd.DataFrame(matriz, columns=self._nombres, index=df.index).astype(int)
        df = df.drop(columns=[self._col]).reset_index(drop=True)
        nuevas = nuevas.reset_index(drop=True)
        resultado = pd.concat([df, nuevas], axis=1)
        print(f"[OK] CodificadorOneHot: '{self._col}' ==> {self._nombres} "
              f"(catálogo fijo: {len(categorias_fit)} categorías)")
        return resultado


################################################
# Escalamiento (dos variantes intercambiables) #
################################################

class EscaladoZScore(Transformador):
    """
    Estandariza columnas a media=0, desviación estándar=1 (z-score).

    Decisión: se aplica a variables con distribución aproximadamente simétrica
    o asimetría moderada (|skewness| < 1.0). StandardScaler es sensible a
    outliers extremos; para escalas discretas acotadas se prefiere MinMax.

    Parámetros
    ----------
    cols : list[str]
        Columnas a estandarizar.
    """

    def __init__(self, cols: list) -> None:
        self._cols = cols
        self._scaler = StandardScaler()

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        faltantes = [c for c in self._cols if c not in df.columns]
        if faltantes:
            raise KeyError(f"EscaladoZScore: columnas no encontradas: {faltantes}")
        df = df.copy()
        df[self._cols] = self._scaler.fit_transform(df[self._cols])
        for col in self._cols:
            print(f"[OK] EscaladoZScore: '{col}' ==> media={df[col].mean():.4f}, "
                  f"std={df[col].std():.4f}")
        return df


class EscaladoMinMax(Transformador):
    """
    Normaliza columnas al rango [0, 1].

    Decisión: se aplica a variables de escala discreta acotada (tipo Likert)
    y a Weekly_GenAI_Hours por su asimetría pronunciada. MinMaxScaler preserva
    la estructura del rango original sin comprimir artificialmente los extremos.

    Parámetros
    ----------
    cols : list[str]
        Columnas a normalizar.
    """

    def __init__(self, cols: list) -> None:
        self._cols = cols
        self._scaler = MinMaxScaler()

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        faltantes = [c for c in self._cols if c not in df.columns]
        if faltantes:
            raise KeyError(f"EscaladoMinMax: columnas no encontradas: {faltantes}")
        df = df.copy()
        df[self._cols] = self._scaler.fit_transform(df[self._cols])
        for col in self._cols:
            print(f"[OK] EscaladoMinMax: '{col}' ==> min={df[col].min():.4f}, "
                  f"max={df[col].max():.4f}")
        return df