
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
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
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo '{file_path}'."
            "Verificar que el archivo CSV se encuentre en data/raw."
        )
    return df


def show_tipos(df: pd.DataFrame) -> None:
    """Muestra dimensiones y tipos de datos del dataset."""
    print(f'Dimensiones: {df.shape}')
    print()
    print(f"{'Columna':<27} {'Tipo Dato'}")
    print("-" * 37)
    print(df.dtypes)
    print()



def show_nulos(df: pd.DataFrame) -> None:
    """Muestra valores nulos por columna."""
    print('Valores nulos por columna:')
    print("-" * 31)
    print(df.isnull().sum())
    print()



def show_estadisticas(df: pd.DataFrame) -> None:
    """Muestra estadísticas descriptivas."""
    print('Estadísticas descriptivas:')
    print("-" * 31)
    print(df.describe())

def mostrar_boxplots(df: pd.DataFrame, columnas: list, titulo: str) -> None:
    """Genera un boxplot por cada columna numérica indicada.

    Parámetros
    ----------
    df       : pd.DataFrame
    columnas : list[str] — columnas numéricas a graficar
    titulo   : str — título general de la figura
    """
    n = len(columnas)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columnas):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7),
                   medianprops=dict(color='orange', linewidth=2))
        ax.set_title(col, fontsize=8, wrap=True)
        ax.tick_params(axis='x', labelbottom=False)
    fig.suptitle(titulo, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def detectar_outliers(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """Detecta valores atípicos por método IQR y retorna tabla resumen.

    Parámetros
    ----------
    df       : pd.DataFrame — dataset original (sin escalar)
    columnas : list[str] — columnas numéricas a analizar

    Retorna
    -------
    pd.DataFrame con Q1, Q3, IQR, límites y conteo de outliers por variable
    """
    resumen = []
    for col in columnas:
        if col not in df.columns:
            continue
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        n_out = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())
        resumen.append({
            'Variable'     : col,
            'Q1'           : round(Q1, 2),
            'Q3'           : round(Q3, 2),
            'IQR'          : round(IQR, 2),
            'Lím. inferior': round(lim_inf, 2),
            'Lím. superior': round(lim_sup, 2),
            'N outliers'   : n_out,
            '% del total'  : round(n_out / len(df) * 100, 2),
        })

    return pd.DataFrame(resumen).set_index('Variable')


def show_categories(df: pd.DataFrame) -> None:
    """Muestra la repetición de las variables por columna.

    Parámetros
    ----------
    df       : pd.DataFrame — dataset original (sin escalar)

    Retorna
    -------
    None
    """
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
      print(f"\n{col}:")
      print(df[col].value_counts().to_string())

def compute_skewness(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Calcula skewness de columnas numéricas y recomienda escalador."""
    resultados = []
    likert_cols = {'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams', 'Tool_Diversity'}

    for col in cols:
        if col not in df.columns:
            print(f'[WARN] Columna {col} no encontrada.')
            continue
        skew_val = df[col].skew()
        abs_skew = abs(skew_val)

        if col in likert_cols:
            interpretacion = 'Escala discreta acotada (dominio definido)'
            escalador = 'MinMaxScaler'
        elif abs_skew < 0.5:
            interpretacion = 'Aproximadamente simétrica'
            escalador = 'StandardScaler'
        elif abs_skew < 1.0:
            interpretacion = 'Asimetría moderada'
            escalador = 'StandardScaler (revisar histograma)'
        else:
            interpretacion = 'Asimetría pronunciada'
            escalador = 'MinMaxScaler'

        resultados.append({
            'Variable':              col,
            'Skewness':              round(skew_val, 4),
            'Interpretación':        interpretacion,
            'Escalador_Recomendado': escalador
        })

    tabla = pd.DataFrame(resultados)
    print('Análisis de skewness — criterio de selección de escalador:')
    print('-' * 75)
    print(tabla.to_string(index=False))
    print()
    return tabla



def drop_id_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Elimina la columna identificadora sin valor predictivo.

    Parámetros
    ----------
    df       : pd.DataFrame — dataset original (sin escalar)
    col      : String con el nombre de la columna

    Retorna
    -------
    pd.DataFram: Datos cargados en un DataFrame
    """
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f'[OK] Columna {col} eliminada. Shape resultante: {df.shape}')
    else:
        print(f'[WARN] Columna {col} no encontrada.')
    return df


def cast_bool_to_int(df: pd.DataFrame, col: str = 'Paid_Subscription') -> pd.DataFrame:
    """Convierte columna booleana a entero (False->0, True->1).

    Parámetros
    ----------
    df  : pd.DataFrame
    col : nombre de la columna booleana (default 'Paid_Subscription')

    Retorna
    -------
    pd.DataFrame con la columna convertida a int64
    """
    if col not in df.columns:
        print(f'[WARN] Columna {col} no encontrada.')
        return df
    print(f'[INFO] {col} — dtype antes: {df[col].dtype}')
    # Conversión via numpy array, evita inferencia de tipos de pandas
    df[col] = df[col].to_numpy().astype('int64')
    print(f'[INFO] {col} — dtype despues: {df[col].dtype}')
    if df[col].dtype != 'int64':
        raise TypeError(
            f'La conversión no produjo int64. dtype resultante: {df[col].dtype}.'
        )
    conteo = df[col].value_counts().sort_index().to_dict()
    print(f'[OK] {col} convertida a int64. Distribución: {conteo}')
    return df

def encode_ordinal(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """Codifica variable ordinal asignando enteros según el orden definido.

    Parámetros
    ----------
    df    : pd.DataFrame
    col   : nombre de la columna a codificar
    order : lista de categorías en orden ascendente
            Ejemplo: ['Low', 'Medium', 'High'] -> {Low:0, Medium:1, High:2}

    Retorna
    -------
    pd.DataFrame con la columna reemplazada por valores enteros
    """
    if col not in df.columns:
        print(f'[WARN] Columna {col} no encontrada.')
        return df
    mapping = {cat: idx for idx, cat in enumerate(order)}
    df[col] = df[col].map(mapping)
    nulos_post = df[col].isnull().sum()
    if nulos_post > 0:
        print(f'[ERROR] {col}: {nulos_post} valores sin mapeo. Verificar order.')
    else:
        print(f'[OK] {col} codificada. Mapping: {mapping}')
    return df


def codificar_one_hot(df: pd.DataFrame, col: str, nombres: list) -> pd.DataFrame:
    """Codifica variable nominal con LabelEncoder + OneHotEncoder (patrón fit/transform).

    Parámetros
    ----------
    df      : pd.DataFrame
    col     : str — columna nominal a codificar
    nombres : list[str] — nombres de las columnas binarias, en orden alfabético
               de las categorías (criterio interno de LabelEncoder)

    Retorna
    -------
    pd.DataFrame con columnas one-hot añadidas y columna original eliminada

    Excepción
    ---------
    KeyError
        Si la columna no existe en el DataFrame.
    ValueError
        Si el número de nombres no coincide con el número de categorías.
    """
    if col not in df.columns:
        raise KeyError(f"Columna '{col}' no encontrada en el DataFrame.")

    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    if len(nombres) != len(le.classes_):
        raise ValueError(
            f"Se esperaban {len(le.classes_)} nombres para '{col}' "
            f"({list(le.classes_)}), pero se recibieron {len(nombres)}."
        )

    datos_le = le.transform(df[col]).reshape(-1, 1)
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(datos_le)
    matriz = ohe.transform(datos_le).toarray()

    nuevas = pd.DataFrame(matriz, columns=nombres, index=df.index).astype(int)
    df = df.drop(columns=[col]).reset_index(drop=True)
    nuevas = nuevas.reset_index(drop=True)
    resultado = pd.concat([df, nuevas], axis=1)
    print(f'[OK] {col} → {nombres}')
    return resultado

def escalar_caracteristicas(df: pd.DataFrame,
                             cols_standard: list,
                             cols_minmax: list):
    """Escala variables numéricas según el escalador recomendado por el análisis de skewness.

    Parámetros
    ----------
    df            : pd.DataFrame
    cols_standard : list[str] — columnas a estandarizar con StandardScaler (z-score)
    cols_minmax   : list[str] — columnas a normalizar con MinMaxScaler (rango [0,1])

    Retorna
    -------
    tuple (pd.DataFrame escalado, StandardScaler ajustado, MinMaxScaler ajustado)

    Excepción
    ---------
    KeyError
        Si alguna de las columnas no existe en el DataFrame.
    """
    faltantes = [c for c in cols_standard + cols_minmax if c not in df.columns]
    if faltantes:
        raise KeyError(f'Columnas no encontradas en el DataFrame: {faltantes}')

    df = df.copy()

    scaler_std = StandardScaler()
    df[cols_standard] = scaler_std.fit_transform(df[cols_standard])
    for col in cols_standard:
        print(f'[OK] {col:<30} ==> StandardScaler | media={df[col].mean():.4f}, std={df[col].std():.4f}')

    scaler_mm = MinMaxScaler()
    df[cols_minmax] = scaler_mm.fit_transform(df[cols_minmax])
    for col in cols_minmax:
        print(f'[OK] {col:<30} ==> MinMaxScaler  | min={df[col].min():.4f}, max={df[col].max():.4f}')

    return df, scaler_std, scaler_mm


def export_processed(df: pd.DataFrame, file_path: str) -> None:
    """Exporta el dataset procesado en formato CSV en la ruta recibida como parámetro

    Parámetros
    ----------
    df        : pd.DataFrame procesado
    file_path : Ruta de salida del archivo CSV
    """
    df.to_csv(file_path, index=False)
    print(f'[OK] Dataset procesado exportado a: {file_path}')
    print(f'     Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas')
    print(f'     Memoria: {df.memory_usage(deep=True).sum() / 1024:.1f} KB')

COLUMNAS_NUMERICAS = [
    'Pre_Semester_GPA',
    'Weekly_GenAI_Hours',
    'Tool_Diversity',
    'Traditional_Study_Hours',
    'Perceived_AI_Dependency',
    'Anxiety_Level_During_Exams',
    'Post_Semester_GPA',
    'Skill_Retention_Score',
]

# Columnas a estandarizar con StandardScaler (distribución simétrica o asimetría moderada)
COLS_STANDARD = [
    'Pre_Semester_GPA',
    'Traditional_Study_Hours',
    'Post_Semester_GPA',
    'Skill_Retention_Score',
]

# Columnas a normalizar con MinMaxScaler (escala discreta acotada o asimetría pronunciada)
COLS_MINMAX = [
    'Weekly_GenAI_Hours',
    'Tool_Diversity',
    'Perceived_AI_Dependency',
    'Anxiety_Level_During_Exams',
]