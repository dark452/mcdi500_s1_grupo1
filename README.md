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