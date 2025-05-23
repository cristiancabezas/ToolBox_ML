import pandas as pd
import numpy as np

dataframe = pd.read_csv("titanic.csv", encoding="utf-8")


def describe_df(dataframe: pd.DataFrame):
    try:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El elemento introducido como dataframe no es un dataframe.")
        if dataframe.empty:
            raise ValueError("El dataframe está vacío.")

        resumen_variables = {}

        total_filas = len(dataframe)

        for columna in dataframe.columns:
            tipo_variable = dataframe[columna].dtype
            porcentaje_nulos = dataframe[columna].isna().mean() * 100 
            valores_unicos = dataframe[columna].nunique(dropna=False)
            cardinalidad = (valores_unicos / total_filas) * 100 if total_filas > 0 else 0 #Evitamos dividir por cero

            resumen_variables[columna] = [
                tipo_variable,
                round(porcentaje_nulos, 2),
                valores_unicos,
                round(cardinalidad, 2)
            ]

        dataframe_salida = pd.DataFrame(resumen_variables, index=[
            "DATA_TYPE", "MISSINGS(%)", "UNIQUE_VALUES", "CARDIN(%)"
        ])
        return dataframe_salida
    except Exception as e:
        print(f"Error en la función describe_df: {e}")


def tipifica_variables(dataframe: pd.DataFrame, umbral_categoria: int, umbral_continua: float):
    try:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El elemento introducido como dataframe no es un dataframe.")
        if not isinstance(umbral_categoria, int):
            raise TypeError("El elemento introducido como umbral_categoria no es un int.")
        if not isinstance(umbral_continua, float):
            raise TypeError("El elemento introducido como umbral_continua no es un float.")
        if dataframe.empty:
            raise ValueError("El dataframe está vacío")
        
        dataframe_salida = []

        total_filas = len(dataframe)

        for columna in dataframe.columns:
            cardinalidad = dataframe[columna].nunique(dropna=False)
            porcentaje_cardinalidad = cardinalidad / total_filas if total_filas > 0 else 0 #Evitamos la división por cero

            if cardinalidad == 2:
                tipo = "Binaria"
            elif cardinalidad < umbral_categoria:
                tipo = "Categórica"
            else:
                if porcentaje_cardinalidad >= umbral_continua:
                    tipo = "Numérica contínua"
                else:
                    tipo = "Numérica discreta"
            
            dataframe_salida.append({
                "variable": columna,
                "tipo_sugerido": tipo
            })
        return pd.DataFrame(dataframe_salida)      
    except Exception as e:
        print(f"Error en la función tipifica_variable: {e}")

print(describe_df(dataframe))

print(tipifica_variables(dataframe, 30, 10.0))