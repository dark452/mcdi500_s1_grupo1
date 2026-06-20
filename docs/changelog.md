# Changelog — MCDI500 Grupo 1 — Impacto de la IA Generativa en Estudiantes

Registro de cambios por fase, construido íntegramente a partir del historial real de commits del
repositorio (`git log --all`). Sigue el formato sugerido por la guía de la Sumativa 4: fecha,
descripción, hash de commit corto y justificación técnica.

Las fechas corresponden al calendario registrado en el repositorio (zona horaria del entorno de
desarrollo del equipo). La sección F4 refleja los commits de la rama `feature/visualizacion`
creados el 2026-06-19; esa rama estaba pendiente de merge a `main` al momento de actualizar
este documento.

---

## [F1] — 2026-06-05 a 2026-06-07

### Added
- Estructura inicial del entorno de trabajo: carpetas `data/raw`, `data/processed`, `notebooks/`,
  `src/`, `docs/`, `.gitignore` y `requirements.txt` (commits `17a3ecf`, `0f5ff33`, `398e014`).
- Dataset crudo `ai_student_impact_dataset.csv` en `data/raw/` como archivo inmutable
  (commit `749e073`): 50.000 registros, 16 columnas.
- `F1_Definicion.ipynb` con descripción del proyecto, carga del dataset y funciones exploratorias
  iniciales: `show_tipos`, `show_nulos`, `show_estadisticas`, `show_categories`
  (commits `7c94cf8`, `3aa226a`, `90dfd85`).
- `README.md` inicial con descripción, estructura del repositorio e instrucciones de reproducción
  (commits `f6fc890`, `c00584a`).

### Changed
- Corrección de ruta del dataset y eliminación de importación de `numpy` no utilizada en
  `F1_Definicion.ipynb` (commit `f1a1c01`).
- Ajuste de funciones de exploración y celdas de comentarios (commit `cb033d9`).
- `.gitignore` ampliado con exclusión de checkpoints de Jupyter y archivos temporales
  (commit `e782afc`).
- `requirements.txt` con versiones fijadas de dependencias del entorno; se agrega `jupyter`
  (commits `08143d8`, `0ee853c`).
- README actualizado con procedimiento de reproducción y librerías (commit `7ff79e0`).

---

## [F2] — 2026-06-08 a 2026-06-10

### Added
- `F2_EDA_Limpieza.ipynb` creado con descripción de la Fase 2 y función `load_data`
  (commit `56e017a`).
- Funciones de exploración y limpieza: `show_tipos`, `show_nulos`, `show_estadisticas`,
  `show_categories`, con excepción robusta en `load_data` (commits `3aa226a`, `90dfd85`,
  `c17433d`).
- Función `compute_skewness` para análisis de asimetría de variables numéricas
  (commit `082e89c`).
- Función `check_outliers_iqr` para detección de outliers con criterio IQR
  (commit `a5542f6`).
- Función `codificar_one_hot` aplicada sobre `Major_Category`, `Primary_Use_Case` e
  `Institutional_Policy` (commit `61e722`).
- Función `escalar_caracteristicas` con StandardScaler y MinMaxScaler (commit `74e3631`).
- Función de exportación del dataset procesado y validación del pipeline con verificaciones
  completas (commits `a8c8963`, `180737`).
- Limpieza y transformación ordinal de datos (commit `f2822fd`).
- Sección F2 añadida al `README.md` con objetivos y funcionalidades (commit `362e3c1`).

### Changed
- Reorganización de secciones, numeración de títulos y mejoras de formato en el notebook
  (commits `67b936c`, `d6ca788`, `d2fe872`).
- Corrección de typo en `load_data` y eliminación de asignación duplicada (commit `0e9278`).
- Corrección de referencia a variable errónea (commit `219b4cb`).
- Eliminación de librería importada duplicada, actualización de tabla informativa
  (commit `3ac39cb`).
- Limpieza de outputs del notebook; corrección de rutas (commit `fc2f1d3`).

---

## [F3] — 2026-06-16 a 2026-06-17

### Added
- `F3_Rendimiento_POO.ipynb` creado con descripción inicial e importación de librerías;
  proyección de F2 incluida (commits `a984989`, `2ae52c5`).
- Módulo `src/functions.py`: refactorización de funciones de F2 a código reutilizable
  (commit `0d401cb`).
- Clase `Transformador` (ABC) con 6 subclases polimórficas: `EliminadorColumna`,
  `CastBooleano`, `CodificadorOrdinal`, `CodificadorOneHot`, `EscaladoZScore`,
  `EscaladoMinMax` (commit `dd59632`).
- Clase `GestorDatos` para encapsular E/S del dataset (commit `0ebde2d`).
- Clase `Preprocesador` con métodos encadenables (`eliminar_id`, `cast_bool`,
  `codificar_ordinales`, `codificar_nominales`, `escalar`, `validar`, `resultado`)
  y clase `Pipeline` como orquestador polimórfico (commit `e1c0843`).
- Sección de validación del pipeline alternativo v1 y v2 en el notebook
  (commits `dbb84d5`, `f976747`).
- Clase `BuscadorPerfilRiesgo`, `OrdenadorMergeSort`, `DetectorOutliersIQR` y
  `AnalizadorRiesgo` (orquestador); sección 5 del notebook (commit `b45342b`).
- Sección 6 de medición de eficiencia con `timeit` (commit `20b8be5`).
- Sección de visualización del análisis (commit `3120230`).
- Dataset procesado `data/processed/` exportado desde F3 (commit `7362879`).
- Diagrama de clases en sección 3 del notebook (commit `fd25c40`).
- Sub-sección markdown 1.3 con importación de librerías (commit `f9ef7e8`).
- Mailmap para normalizar nombres de autores en el historial (commits `ab3286f`, `0d17e08`).
- README actualizado con sección Fase 3 (commit `fee0cc1`).

### Changed
- README actualizado con diagrama de clases y limitaciones conocidas (commit `0d0cf7c`).
- README actualizado con instrucciones de reproducción del entorno (commit `7bbe76e`).
- `.gitignore` actualizado para incluir `data/processed/` y excluir `__pycache__`
  (commits `898ab28`, `46664e5`, `ed469ac`).
- Eliminación de librerías no utilizadas en `src/` (commit `3c6ac0a`).
- Registro de resultados del notebook (timeit y análisis) en dos iteraciones
  (commits `b8e5207`, `cdae102`).

### Fixed
- Bug en `CodificadorOneHot`: el método fallaba al procesar lotes parciales porque el catálogo
  de categorías no estaba fijado; se resolvió definiendo el conjunto de categorías al momento
  de la primera transformación (commit `d9840b0`).
- Pipeline alternativo sección 3.3 del notebook (commit `918dae3`).

---

## [F4] — 2026-06-19

### Added
- `notebooks/F4_Integrador.ipynb` creado como notebook integrador del cierre de proyecto
  (commit `287d73f`).
- Configuraciones iniciales del notebook y resumen de la Fase 1 con hipótesis formalizada
  (commit `987421a`).
- Resumen de Fase 2, visualizaciones de los tres actos del storytelling y resumen de Fase 3
  (commit `47a5fca`).
- Secciones de resultados, validación y reflexión técnica, discusión, conclusiones y tabla de
  trazabilidad de mejoras F1→F4; exportación del dataset procesado final
  (commit `472c455`).
- Figuras de los tres actos exportadas a `data/processed/`:
  `F4_acto1_contexto.png`, `F4_acto2_conflicto.png`, `F4_acto3_resolucion.png`
  (commit `721cdd4`).
- Dataset procesado final exportado: `data/processed/ai_student_impact_processed_f4.csv`
  (50.001 filas incluyendo encabezado) (commit `721cdd4`).

### Changed
- Celdas markdown del notebook corregidas y ajustadas tras la ejecución completa del pipeline
  integrador (commit `721cdd4`).