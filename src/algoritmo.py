"""
algoritmos.py
=============
Núcleo algorítmico de la Fase 3.

Problema definido: ¿Qué estudiantes presentan el perfil de mayor riesgo académico,
combinando alto nivel de burnout (Burnout_Risk_Level=2), bajo rendimiento
post-semestre (Post_Semester_GPA en el cuartil inferior) y uso intensivo de IA
(Weekly_GenAI_Hours en el cuartil superior)?

Módulos implementados
---------------------
1. BuscadorPerfilRiesgo  — búsqueda lineal y binaria (con medición de complejidad).
2. OrdenadorMergeSort    — merge sort recursivo sobre columna numérica.
3. DetectorOutliersIQR   — detección de valores atípicos por método IQR.
4. AnalizadorRiesgo      — clase orquestadora que integra los tres módulos anteriores.


"""

from __future__ import annotations

import timeit
import numpy as np
import pandas as pd
from typing import Optional


#####################################
# 1. Búsqueda de perfiles de riesgo #
#####################################

class BuscadorPerfilRiesgo:
    """
    Busca estudiantes con perfil de alto riesgo académico.

    Criterio de riesgo:
        - Burnout_Risk_Level = 2 (High)
        - Post_Semester_GPA  ≤ umbral_gpa (por defecto: percentil 25 del dataset)
        - Weekly_GenAI_Hours ≥ umbral_ia  (por defecto: percentil 75 del dataset)

    Implementa búsqueda lineal O(n) y búsqueda binaria O(log n) sobre el GPA
    ordenado, midiendo el costo de cada estrategia con timeit.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset procesado (post-preprocesamiento).
    umbral_gpa : float, optional
        Umbral de GPA bajo (ya estandarizado). Por defecto: percentil 25.
    umbral_ia : float, optional
        Umbral de horas de IA altas (ya normalizadas [0,1]). Por defecto: percentil 75.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        umbral_gpa: Optional[float] = None,
        umbral_ia: Optional[float] = None,
    ) -> None:
        self._df = df.copy()
        self._umbral_gpa = umbral_gpa if umbral_gpa is not None \
            else float(df["Post_Semester_GPA"].quantile(0.25))
        self._umbral_ia  = umbral_ia  if umbral_ia  is not None \
            else float(df["Weekly_GenAI_Hours"].quantile(0.75))

    # *** Búsqueda lineal O(n) ***

    def busqueda_lineal(self) -> pd.DataFrame:
        """
        Recorre el dataset fila por fila buscando el perfil de riesgo.

        Complejidad temporal: O(n) — se revisa cada registro una sola vez.

        Retorna
        -------
        pd.DataFrame
            Subconjunto de filas que cumplen los tres criterios de riesgo.
        """
        indices = []
        for idx, fila in self._df.iterrows():
            if (
                fila["Burnout_Risk_Level"] == 2
                and fila["Post_Semester_GPA"] <= self._umbral_gpa
                and fila["Weekly_GenAI_Hours"] >= self._umbral_ia
            ):
                indices.append(idx)
        return self._df.loc[indices].copy()

    # *** Búsqueda binaria O(log n) ***

    def busqueda_binaria_gpa(self, lista_gpa_ordenada: list, objetivo: float) -> int:
        """
        Búsqueda binaria sobre lista de GPA ordenada ascendentemente.

        Devuelve el índice del primer elemento ≤ objetivo usando bisección.
        Complejidad temporal: O(log n).

        Parámetros
        ----------
        lista_gpa_ordenada : list[float]
            Lista de valores de GPA en orden ascendente.
        objetivo : float
            Umbral de búsqueda.

        Retorna
        -------
        int
            Cantidad de elementos ≤ objetivo (posición de inserción).
        """
        izq, der = 0, len(lista_gpa_ordenada) - 1
        posicion = -1
        while izq <= der:
            medio = (izq + der) // 2
            if lista_gpa_ordenada[medio] <= objetivo:
                posicion = medio
                izq = medio + 1
            else:
                der = medio - 1
        return posicion + 1   # cantidad de elementos ≤ objetivo

    # *** Comparación de tiempos ***

    def comparar_tiempos(self, n_repeticiones: int = 3) -> dict:
        """
        Mide y compara el tiempo de la búsqueda lineal vs. la operación
        vectorizada equivalente en pandas.

        Parámetros
        ----------
        n_repeticiones : int
            Número de repeticiones para timeit (default 3).

        Retorna
        -------
        dict
            {'lineal_s': float, 'vectorizada_s': float, 'factor': float}
        """
        t_lineal = timeit.timeit(
            lambda: self.busqueda_lineal(), number=n_repeticiones
        ) / n_repeticiones

        def vectorizada():
            mask = (
                (self._df["Burnout_Risk_Level"] == 2)
                & (self._df["Post_Semester_GPA"] <= self._umbral_gpa)
                & (self._df["Weekly_GenAI_Hours"] >= self._umbral_ia)
            )
            return self._df[mask]

        t_vector = timeit.timeit(vectorizada, number=n_repeticiones) / n_repeticiones

        factor = t_lineal / t_vector if t_vector > 0 else float("inf")
        resultado = {
            "lineal_s":      round(t_lineal, 6),
            "vectorizada_s": round(t_vector, 6),
            "factor":        round(factor, 1),
        }
        print(f"[INFO] Búsqueda lineal:     {t_lineal:.6f} s")
        print(f"[INFO] Búsqueda vectorizada:{t_vector:.6f} s")
        print(f"[INFO] La vectorizada es ≈{factor:.1f}× más rápida.")
        print("[INFO] Complejidad: lineal O(n·iterrows) vs. vectorizada O(n) en C-level.")
        return resultado


###########################
# 2. Merge Sort recursivo #
###########################

class OrdenadorMergeSort:
    """
    Implementa Merge Sort recursivo (divide y vencerás) sobre una columna numérica.

    Complejidad temporal: O(n log n) en todos los casos.
    Complejidad espacial: O(n) por las listas auxiliares en la combinación.

    Se utiliza para ordenar estudiantes por Post_Semester_GPA y comparar
    el costo de la versión recursiva frente a sorted() nativo de Python.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset del cual se extraerá la columna a ordenar.
    col : str
        Nombre de la columna numérica a ordenar.
    """

    def __init__(self, df: pd.DataFrame, col: str = "Post_Semester_GPA") -> None:
        self._df  = df
        self._col = col

    # *** Algoritmo recursivo ***

    def _combinar(self, izq: list, der: list) -> list:
        """Combina dos sublistas ordenadas en una sola lista ordenada. O(n)."""
        resultado, i, j = [], 0, 0
        while i < len(izq) and j < len(der):
            if izq[i] <= der[j]:
                resultado.append(izq[i]); i += 1
            else:
                resultado.append(der[j]); j += 1
        resultado.extend(izq[i:])
        resultado.extend(der[j:])
        return resultado

    def merge_sort(self, lista: list) -> list:
        """
        Ordena una lista de números con Merge Sort recursivo.

        Caso base : len(lista) <= 1 → ya está ordenada.
        Caso recursivo: divide a la mitad, ordena cada parte, combina.

        Parámetros
        ----------
        lista : list[float]
            Lista de valores a ordenar.

        Retorna
        -------
        list[float]
            Lista ordenada ascendentemente.
        """
        if len(lista) <= 1:          # CASO BASE
            return lista
        medio = len(lista) // 2
        izq = self.merge_sort(lista[:medio])   # RECURSIÓN izquierda
        der = self.merge_sort(lista[medio:])   # RECURSIÓN derecha
        return self._combinar(izq, der)        # COMBINAR

    def ordenar_columna(self) -> list:
        """Extrae la columna y la ordena con merge_sort."""
        datos = self._df[self._col].dropna().tolist()
        return self.merge_sort(datos)

    # *** Comparación de tiempos ***

    def comparar_tiempos(self, n_repeticiones: int = 3) -> dict:
        """
        Compara merge_sort recursivo vs. sorted() nativo de Python.

        Retorna
        -------
        dict
            {'merge_sort_s': float, 'sorted_s': float, 'factor': float}
        """
        datos = self._df[self._col].dropna().tolist()

        t_merge = timeit.timeit(
            lambda: self.merge_sort(datos), number=n_repeticiones
        ) / n_repeticiones

        t_sorted = timeit.timeit(
            lambda: sorted(datos), number=n_repeticiones
        ) / n_repeticiones

        factor = t_merge / t_sorted if t_sorted > 0 else float("inf")
        resultado = {
            "merge_sort_s": round(t_merge, 6),
            "sorted_s":     round(t_sorted, 6),
            "factor":       round(factor, 1),
        }
        print(f"[INFO] Merge Sort recursivo: {t_merge:.6f} s")
        print(f"[INFO] sorted() nativo:      {t_sorted:.6f} s")
        print(f"[INFO] sorted() es ≈{factor:.1f}× más rápido (implementado en C).")
        print("[INFO] Ambos son O(n log n); la diferencia es el overhead de Python puro.")
        return resultado


################################
# 3. Detección de outliers IQR #
################################

class DetectorOutliersIQR:
    """
    Detecta valores atípicos en variables numéricas usando el método IQR.

    Un valor es atípico si cae fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
    Complejidad: O(n) por columna (un recorrido para calcular cuartiles).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset a analizar.
    columnas : list[str]
        Columnas numéricas a inspeccionar.
    """

    def __init__(self, df: pd.DataFrame, columnas: list) -> None:
        self._df = df
        self._columnas = columnas

    def detectar(self) -> pd.DataFrame:
        """
        Calcula estadísticos IQR y cuenta outliers por columna.

        Retorna
        -------
        pd.DataFrame
            Tabla resumen con Q1, Q3, IQR, límites y conteo de outliers.
        """
        resumen = []
        for col in self._columnas:
            if col not in self._df.columns:
                print(f"[WARN] DetectorOutliersIQR: columna '{col}' no encontrada.")
                continue
            q1  = self._df[col].quantile(0.25)
            q3  = self._df[col].quantile(0.75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            n_out = int(
                ((self._df[col] < lim_inf) | (self._df[col] > lim_sup)).sum()
            )
            resumen.append({
                "Variable":      col,
                "Q1":            round(q1, 4),
                "Q3":            round(q3, 4),
                "IQR":           round(iqr, 4),
                "Lím. inferior": round(lim_inf, 4),
                "Lím. superior": round(lim_sup, 4),
                "N outliers":    n_out,
                "% del total":   round(n_out / len(self._df) * 100, 2),
            })
        return pd.DataFrame(resumen).set_index("Variable")

    def perfil_outliers(self) -> pd.DataFrame:
        """
        Devuelve solo las filas con al menos un valor atípico en alguna columna.

        Retorna
        -------
        pd.DataFrame
            Subconjunto de filas con outliers.
        """
        mask = pd.Series(False, index=self._df.index)
        for col in self._columnas:
            if col not in self._df.columns:
                continue
            q1  = self._df[col].quantile(0.25)
            q3  = self._df[col].quantile(0.75)
            iqr = q3 - q1
            mask |= (self._df[col] < q1 - 1.5 * iqr) | \
                    (self._df[col] > q3 + 1.5 * iqr)
        return self._df[mask].copy()


#########################################
# 4. Analizador de riesgo (orquestador) #
#########################################

class AnalizadorRiesgo:
    """
    Orquesta los tres módulos algorítmicos para responder la pregunta de investigación:

    ¿Qué características definen a los estudiantes con mayor riesgo académico
    (burnout alto + GPA bajo + uso intensivo de IA) y cómo se distribuyen?

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset procesado (post-Preprocesador).
    cols_numericas : list[str]
        Columnas numéricas para análisis de outliers.
    """

    _COLS_NUMERICAS_DEFAULT = [
        "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
        "Traditional_Study_Hours", "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams", "Post_Semester_GPA", "Skill_Retention_Score"
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        cols_numericas: Optional[list] = None,
    ) -> None:
        self._df = df.copy()
        self._cols_num = cols_numericas or self._COLS_NUMERICAS_DEFAULT

        # Instancia los tres módulos (composición)
        self._buscador  = BuscadorPerfilRiesgo(df)
        self._ordenador = OrdenadorMergeSort(df)
        self._detector  = DetectorOutliersIQR(df, self._cols_num)

    def ejecutar_analisis_completo(self) -> dict:
        """
        Ejecuta los tres algoritmos y devuelve un diccionario de resultados.

        Retorna
        -------
        dict
            {
                'perfil_riesgo':    pd.DataFrame,
                'gpa_ordenado':     list,
                'resumen_outliers': pd.DataFrame,
                'tiempos_busqueda': dict,
                'tiempos_orden':    dict,
            }
        """
        print("=" * 60)
        print("ANÁLISIS DE RIESGO ACADÉMICO — Fase 3")
        print("=" * 60)

        print("\n[1/5] Búsqueda de perfil de riesgo (lineal)...")
        perfil = self._buscador.busqueda_lineal()
        print(f"      → {len(perfil)} estudiantes identificados.")

        print("\n[2/5] Comparando tiempos de búsqueda...")
        t_busqueda = self._buscador.comparar_tiempos(n_repeticiones=3)

        print("\n[3/5] Ordenando GPA con Merge Sort recursivo...")
        gpa_ord = self._ordenador.ordenar_columna()
        print(f"      → {len(gpa_ord)} valores ordenados. "
              f"Rango: [{gpa_ord[0]:.4f}, {gpa_ord[-1]:.4f}]")

        print("\n[4/5] Comparando tiempos de ordenamiento...")
        t_orden = self._ordenador.comparar_tiempos(n_repeticiones=3)

        print("\n[5/5] Detectando outliers por IQR...")
        resumen_out = self._detector.detectar()
        print(resumen_out.to_string())

        print("\n[OK] Análisis completo.")

        return {
            "perfil_riesgo":    perfil,
            "gpa_ordenado":     gpa_ord,
            "resumen_outliers": resumen_out,
            "tiempos_busqueda": t_busqueda,
            "tiempos_orden":    t_orden,
        }

    def __repr__(self) -> str:
        return f"AnalizadorRiesgo(n={len(self._df)}, cols={len(self._cols_num)})"
