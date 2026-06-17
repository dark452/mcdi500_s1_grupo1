"""
preprocesador.py
================
Clase Preprocesador que encapsula el pipeline completo de la Fase 2
refactorizado a POO, y clase Pipeline que orquesta las etapas via polimorfismo.

Principios aplicados
--------------------
- Encapsulamiento : estado interno en self._df (protegido con guion bajo).
- Herencia        : Preprocesador hereda de object; Pipeline opera sobre Transformador.
- Polimorfismo    : Pipeline llama .aplicar(df) sin importar la subclase concreta.
- Responsabilidad única: cada método realiza una sola transformación conceptual.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

from transformador import (
    Transformador,
    EliminadorColumna,
    CastBooleano,
    CodificadorOrdinal,
    CodificadorOneHot,
    EscaladoZScore,
    EscaladoMinMax,
)


############################################
# Pipeline orquestador (polimorfismo puro) #
############################################

class Pipeline:
    """
    Orquesta una lista de etapas :class:`Transformador` en orden.

    El Pipeline no sabe qué hace cada etapa: las llama todas por igual via
    :método:`Transformador.aplicar`. Eso es polimorfismo.

    Parámetros
    ----------
    etapas : list[Transformador]
        Transformaciones a aplicar en orden.

    Ejemplos
    --------
    >>> pipe = Pipeline([EliminadorColumna("Student_ID"), CastBooleano()])
    >>> df_out = pipe.ejecutar(df)
    """

    def __init__(self, etapas: List[Transformador]) -> None:
        self._etapas = etapas

    def ejecutar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta todas las etapas en orden.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset de entrada.

        Retorno
        -------
        pd.DataFrame
            Dataset completamente transformado.
        """
        print(f"[INFO] Pipeline iniciado — {len(self._etapas)} etapas.")
        for i, etapa in enumerate(self._etapas, 1):
            print(f"\n  [{i}/{len(self._etapas)}] {etapa}")
            df = etapa.aplicar(df)
        print(f"\n[OK] Pipeline completado. Shape final: {df.shape}")
        return df

    def __repr__(self) -> str:
        etapas_str = ", ".join(str(e) for e in self._etapas)
        return f"Pipeline([{etapas_str}])"


##############################################
# Preprocesador: encapsula el pipeline de F2 #
##############################################

class Preprocesador:
    """
    Encapsula el pipeline de preprocesamiento de la Fase 2 en una clase.

    El estado interno (_df) es protegido; los métodos lo modifican y devuelven
    self para permitir el encadenamiento fluido de operaciones.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset raw de entrada. Se trabaja sobre una copia para preservar el original.

    Ejemplos
    --------
    >>> prep = Preprocesador(df_raw)
    >>> df_procesado = (prep
    ...     .eliminar_id()
    ...     .cast_bool()
    ...     .codificar_ordinales()
    ...     .codificar_nominales()
    ...     .escalar()
    ...     .resultado())
    """

    # Configuración de columnas (constantes de clase)
    _COL_ID   = "Student_ID"
    _COL_BOOL = "Paid_Subscription"

    _ORDINALES = {
        "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
        "Burnout_Risk_Level": ["Low", "Medium", "High"],
    }

    _NOMINALES = {
        "Major_Category": {
            "categorias": ["Arts", "Business", "Humanities", "Medical", "STEM"],
            "nombres": [
                "major_Arts", "major_Business", "major_Humanities",
                "major_Medical", "major_STEM"
            ],
        },
        "Primary_Use_Case": {
            "categorias": [
                "Copywriting/Drafting", "Debugging/Troubleshooting",
                "Direct_Answer_Generation", "Ideation", "Summarizing_Reading"
            ],
            "nombres": [
                "use_Copywriting_Drafting", "use_Debugging_Troubleshooting",
                "use_Direct_Answer_Generation", "use_Ideation", "use_Summarizing_Reading"
            ],
        },
        "Institutional_Policy": {
            "categorias": [
                "Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"
            ],
            "nombres": [
                "policy_Actively_Encouraged", "policy_Allowed_With_Citation",
                "policy_Strict_Ban"
            ],
        },
    }

    _COLS_STANDARD = [
        "Pre_Semester_GPA", "Traditional_Study_Hours",
        "Post_Semester_GPA", "Skill_Retention_Score"
    ]

    _COLS_MINMAX = [
        "Weekly_GenAI_Hours", "Tool_Diversity",
        "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()   # encapsulamiento: copia protegida

    # *** Métodos de transformación (cada uno: una sola responsabilidad) ***

    def eliminar_id(self) -> "Preprocesador":
        """Elimina Student_ID: identificador sin valor predictivo."""
        self._df = EliminadorColumna(self._COL_ID).aplicar(self._df)
        return self

    def cast_bool(self) -> "Preprocesador":
        """Convierte Paid_Subscription bool → int64."""
        self._df = CastBooleano(self._COL_BOOL).aplicar(self._df)
        return self

    def codificar_ordinales(self) -> "Preprocesador":
        """Codifica las tres variables ordinales con mapeo explícito."""
        for col, orden in self._ORDINALES.items():
            self._df = CodificadorOrdinal(col, orden).aplicar(self._df)
        return self

    def codificar_nominales(self) -> "Preprocesador":
        """Aplica One-Hot Encoding a las tres variables nominales, usando un
        catálogo de categorías fijo para que el resultado sea consistente sin 
        importar cuántas filas se procesen."""
        for col, spec in self._NOMINALES.items():
            self._df = CodificadorOneHot(
                col, spec["nombres"], categorias=spec["categorias"]
            ).aplicar(self._df)
        return self

    def escalar(self) -> "Preprocesador":
        """
        Escala variables numéricas según su distribución (determinada en F2
        por análisis de skewness).

        - StandardScaler (z-score): distribución simétrica o asimetría moderada.
        - MinMaxScaler ([0,1]): escala discreta acotada o asimetría pronunciada.
        """
        self._df = EscaladoZScore(self._COLS_STANDARD).aplicar(self._df)
        self._df = EscaladoMinMax(self._COLS_MINMAX).aplicar(self._df)
        return self

    def resultado(self) -> pd.DataFrame:
        """Devuelve el DataFrame transformado."""
        return self._df.copy()

    # *** Validación interna ***

    def validar(self) -> "Preprocesador":
        """
        Valida el estado actual del dataset con asserts.
        Lanza AssertionError si alguna condición no se cumple.
        """
        df = self._df

        assert df.isnull().sum().sum() == 0, "Validación fallida: hay valores nulos."
        assert df.duplicated().sum() == 0,   "Validación fallida: hay filas duplicadas."

        no_num = [c for c in df.columns
                  if not pd.api.types.is_numeric_dtype(df[c])]
        assert not no_num, f"Columnas no numéricas: {no_num}"

        assert df.shape[1] == 25, (
            f"Se esperaban 25 columnas, hay {df.shape[1]}."
        )
        ohe_cols = [c for c in df.columns
                    if c.startswith(("major_", "use_", "policy_"))]
        for col in ohe_cols:
            assert set(df[col].unique()).issubset({0, 1}), \
                f"OHE inválido en '{col}': valores fuera de {{0,1}}."

        assert set(df["Burnout_Risk_Level"].unique()).issubset({0, 1, 2}), \
            "Burnout_Risk_Level tiene valores inesperados."

        print("[OK] Validación completa: dataset consistente.")
        return self

    def __repr__(self) -> str:
        return f"Preprocesador(shape={self._df.shape})"