# MCDI500 - Impacto de la IA Generativa en Estudiantes

## Grupo 1 | MCDI500 - Programación para la ciencia de datos

### Integrantes

- Pablo Ignacio Balbontin Constenla @pabbalbontin-maker
- Melany Esmeralda Reyes Leiva @melanyreyesy
- Ingeborg Andrea Munoz Carnot @dark452
- Mario Alejandro Lopez Pulgar @malp2203

### Descripción de problemática

El objetivo del proyecto es realizar un análisis del impacto de la utilización y frecuencia  de uso de la IA generativa sobre el rendimiento académico y el nivel de agotamiento de estudiantes universitarios.

*Dataset (AI student impact)*:

- 50.000 registros
- 16 columnas

*Columnas de interés*:

- Post_Semester_GPA (rendimiento académico)
- Burnout_Risk_Level (nivel de agotamiento de los estudiantes universitarios)

*Estadísticas descriptivas del dataset*:


|       |    Student_ID |   Pre_Semester_GPA |   Weekly_GenAI_Hours |   Tool_Diversity |   Traditional_Study_Hours |   Perceived_AI_Dependency |   Anxiety_Level_During_Exams |   Post_Semester_GPA |   Skill_Retention_Score |
|:------|--------------:|-------------------:|---------------------:|-----------------:|--------------------------:|--------------------------:|-----------------------------:|--------------------:|------------------------:|
| count |  50000.000000 |       50000.000000 |         50000.000000 |     50000.000000 |              50000.000000 |              50000.000000 |                 50000.000000 |        50000.000000 |            50000.000000 |
| mean  | 125000.500000 |           3.146102 |             8.427752 |         2.800260 |                 11.209271 |                  3.505360 |                     4.270760 |            3.349299 |               75.798125 |
| std   |  14433.901067 |           0.478854 |             8.269490 |         1.188020 |                  5.156426 |                  1.820812 |                     2.144066 |            0.495673 |               13.281626 |
| min   | 100001.000000 |           1.183000 |             0.000000 |         1.000000 |                  1.000000 |                  1.000000 |                     1.000000 |            1.000000 |               10.780000 |
| 25%   | 112500.750000 |           2.834000 |             2.390000 |         2.000000 |                  7.560000 |                  2.000000 |                     3.000000 |            3.023750 |               66.820000 |
| 50%   | 125000.500000 |           3.210000 |             5.800000 |         3.000000 |                 11.180000 |                  3.000000 |                     4.000000 |            3.421000 |               76.000000 |
| 75%   | 137500.250000 |           3.521000 |            11.720000 |         4.000000 |                 14.710000 |                  5.000000 |                     6.000000 |            3.749000 |               85.190000 |
| max   | 150000.000000 |           3.998000 |            40.000000 |         5.000000 |                 35.860000 |                 10.000000 |                    10.000000 |            4.000000 |              100.000000 |

### Estructura del repositorio

```bash
mcdi500_s1_grupo1/
├── data/raw/          # dataset original (INMUTABLE)
├── data/processed/    # datos transformados
├── notebooks/         # notebooks Jupyter por fase
├── src/               # funciones Python reutilizables
├── docs/              # documentacion adicional
├── requirements.txt
└── README.md
```

### Cómo reproducir el entorno

El siguiente procedimiento, permite clonar el repositorio remoto en un entorno local.

```bash
git clone https://github.com/dark452/mcdi500_s1_grupo1.git
cd mcdi500_s1_grupo1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip freeze > requirements.txt
jupyter notebook notebooks/F1_Definicion.ipynb
```

### Fase 2 — EDA y Preparación de Datos

*Proyecto:* Impacto de la IA Generativa en Estudiantes de Educación Superior  

#### Objetivo

Explorar el dataset en profundidad, limpiar y transformar los datos para dejarlos listos para la etapa de modelado.

#### Funcionalidades

1. *Limpieza de datos*: Tratamiento de inconsistencias y valores atípicos si se identifican.
2. *Codificación de variables categóricas*: Transformar variables nominales y ordinales a formato numérico.
3. *Escalamiento de variables numéricas*: Normalizar o estandarizar según la distribución de cada variable.
4. *Exportación*: Guardar los datos transformados en data/processed/ para uso en F3.

#### Librerías utilizadas

Se importan las librerías necesarias para cargar, explorar, transformar y visualizar el dataset.

| Librería | Función |
|---|---|
| `pandas` | Cargar y manipular la tabla de datos (el `DataFrame`). |
| `numpy` | Operaciones numéricas y fijación de semilla aleatoria. |
| `matplotlib` / `seaborn` | Visualizaciones del EDA (histogramas, boxplots, heatmap). |
| `sklearn.preprocessing` | Las herramientas de codificación (`LabelEncoder`, `OneHotEncoder`) y escalamiento de variables numéricas (`StandardScaler`, `MinMaxScaler`). |

`np.random.seed(42)` fija la semilla aleatoria: garantiza que cualquier proceso con azar dé **siempre el mismo resultado**, asegurando la *reproducibilidad* exigida en la fase.

#### Funciones para reutilizar en el código Python

* **`load_data(file_path)`**: Carga el dataset desde un archivo CSV y retorna un DataFrame con los datos.
* **`show_tipos(df)`**: Muestra las columnas del dataset y el tipo de dato de cada una.
* **`show_nulos(df)`**: Muestra la cantidad de valores nulos por columna.
* **`show_estadisticas(df)`**: Muestra estadísticas descriptivas del dataset.
* **`show_categories(df)`**: Muestra la frecuencia de valores por columna categórica.
* **`mostrar_boxplots(df, columnas, titulo)`**: Genera un boxplot por cada columna numérica recibida por parámetro.
* **`detectar_outliers(df, columnas)`**: Detectar valores atípicos mediante el método IQR y retorna una tabla resumen.
* **`compute_skewness(df, cols)`**: Calcula el *skewness* de columnas numéricas, recomienda el escalador adecuado y retorna un DataFrame con los resultados.
* **`drop_id_column(df, col)`**: Elimina del DataFrame la columna recibida por parámetro.
* **`cast_bool_to_int(df, col)`**: Convierte una columna booleana a entero (`False` → 0, `True` → 1) y retorna el DataFrame con la columna en tipo `int64`.
* **`encode_ordinal(df, col, order)`**: Codifica una variable ordinal asignando enteros según el orden definido y retorna el DataFrame con la columna reemplazada.
* **`codificar_one_hot(df, col, nombres)`**: Codifica una variable nominal mediante `LabelEncoder` y `OneHotEncoder`, retornando el DataFrame con las columnas binarias añadidas y la columna original eliminada.
* **`escalar_caracteristicas(df, cols_standard, cols_minmax)`**: Escala las variables numéricas aplicando `StandardScaler` o `MinMaxScaler` según la distribución de cada columna, y retorna una tupla con el DataFrame escalado y ambos escaladores ajustados.
* **`export_processed(df, file_path)`**: Exporta el dataset procesado en formato CSV en la ruta recibida por parámetro.
  


