import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


dataframe = pd.read_csv("titanic.csv", encoding="utf-8")


def describe_df(dataframe: pd.DataFrame):
    """
    Genera un resumen descriptivo de las variables de un DataFrame.

    Recibiremos información sobre el dataset:
        - Tipo de dato
        - Porcentaje de nulos
        - Número de valores únicos 
        - Cardinalidad relativa 

    Parámetros
    ----------
    dataframe : pd.DataFrame
        Dataset a usar

    Retorna
    -------
    pd.DataFrame
        Un nuevo DataFrame con las siguientes filas:
        - 'DATA_TYPE': tipo de dato de cada columna
        - 'MISSINGS(%)': porcentaje de valores nulos por columna
        - 'UNIQUE_VALUES': número de valores únicos (incluyendo NaN)
        - 'CARDIN(%)': cardinalidad relativa en porcentaje

    Excepciones
    ------
    TypeError
        Si el objeto proporcionado no es un dataset
    ValueError
        Si el dataset está vacío
    """

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
    """
    Clasifica las variables de un dataset en diferentes tipos.

    Analiza cada columna del dataset y ofrece una tipificación a través de la cardinalidad obtenida,
    el porcentaje de la cardinalidad sobre el total de la fila. Las columnas se van a clasificar en:
    Binaria, Categórica, Numérica discreta o Numérica contínua.

    
    Parámetros
    ----------
    dataframe : pd.DataFrame
        DataFrame que se desea analizar
    
    umbral_categoria : int
        Número máximo de valores únicos para clasificar una variable como categórica
    
    umbral_continua : float
        Porcentaje mínimo de cardinalidad para clasificar una variable como numérica continua

    Retorna
    -------
    pd.DataFrame
        DataFrame con dos columnas:
        - variable: nombre de la columna original
        - tipo_sugerido: tipo estadístico sugerido según los criterios definidos

    Excepciones
    -----------
    TypeError
        Si los argumentos no son del tipo esperado

    ValueError
        Si el dataset está vacío
    """

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









def get_features_num_regression(data, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de columnas que tienen una correlación con el 'target_col'.

    A paritr de las variables numéricas, hace la correlación con la variable 'target_col', si cumple 
    con que sea superior en valor absoluto al umbral_corr, y si hay un 'pvalue', también se usa
    para las correlaciones.

    Parámetros
    ----------
    data : DataFrame
        Es el DataFrame con todas las variables contanto la 'target_col'
        
    target_col : str
        Es el nombre de la columna target o valor a predecir
        
    umbral_corr : float
        Umbral mínimo de correlación. Valores entre 1 y 0
        
    pvalue : float
        Nivel de satisfacción estadística. Valores entre 1 y 0

    Retorna
    -------
    lista : list
        Una lista con nombres de columnas que cumplen con lo exigido de correlación y p-valor

    Excepciones
    ----------
    TypeError
        Si los valores introducidos no cumplen con lo requerido
        
    ValueError
        Si 'target_col' no se encuentra en el DataFrame
    
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError('El argumento "data" tiene que ser un Pandas DataFrame')
            
        if not np.issubdtype(data[target_col].dtype, np.number):
            raise TypeError('El argumento "target_col" tiene que ser numérica')

        if data[target_col].nunique() < 10:
            raise TypeError('La variable "target_col" tiene que ser numérica continua o discreta')

        if target_col not in data.columns:
            raise ValueError('El argumento "target_col" tiene que pertenecer al DataFrame')

        if not isinstance(umbral_corr, float) and not (0 <= umbral_corr >= 1):
            raise TypeError('El argumento "umbral_corr" tiene que ser un float entre 0 y 1')

        if pvalue != None:
            if not isinstance(pvalue, float) and not (0 <= pvalue >= 1):
                raise TypeError('El "pvalue" tiene que ser un float entre 0 y 1')
                return None


        numericas = data.select_dtypes(include= np.number).columns
        numericas = numericas.drop(target_col)

        lista = []
        for x in numericas:
            datos = data[[x, target_col]].dropna()
            corr, p = pearsonr(datos[x], datos[target_col])
            if abs(corr) < umbral_corr:
                if pvalue is None or p <= (1 - pvalue):
                    lista.append(x)
        return lista
    except Exception as e:
        print(f"Error en la función get_features_num_regression: {e}")
        return None
    



def plot_features_num_regression(data, target_col = '', columns = [], umbral_corr = 0, pvalue = None):
    """
    Devuelve una lista de columnas que tienen una correlación con el 'target_col'.

    A paritr de las variables numéricas, hace la correlación con la variable 'target_col', si cumple 
    con que sea superior en valor absoluto al umbral_corr, y si hay un 'pvalue', también se usa
    para las correlaciones.

    También, devuelve un pairplot.
    
    Parámetros
    ----------
    data : DataFrame
        Es el DataFrame con todas las variables contanto la 'target_col'
        
    target_col : str
        Es el nombre de la columna target o valor a predecir

    columns : list
        Es una lista que puede estár vacía
        
    umbral_corr : float
        Umbral mínimo de correlación. Valores entre 1 y 0
        
    pvalue : float
        Nivel de satisfacción estadística. Valores entre 1 y 0

    Retorna
    -------
    lista : list
        Una lista con nombres de columnas que cumplen con lo exigido de correlación y p-valor
        
    pairplot : seaborn
        Genera un pairplot con los resultados para visualizarlos

    Excepciones
    ----------
    TypeError
        Si los valores introducidos no cumplen con lo requerido
        
    ValueError
        Si 'target_col' no se encuentra en el DataFrame
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError('El argumento "data" tiene que ser un Pandas DataFrame')

        if not np.issubdtype(data[target_col].dtype, np.number):
            raise TypeError('El argumento "target_col" tiene que ser numérica')

        if data[target_col].nunique() < 10:
            return'La variable "target_col" tiene que ser numérica continua o discreta'

        if target_col not in data.columns:
            raise ValueError('El argumento "target_col" tiene que pertenecer al DataFrame')
            
        if not isinstance(umbral_corr, float) and not (0 <= umbral_corr >= 1):
            raise TypeError('El argumento "umbral_corr" tiene que ser un float entre 0 y 1')

        if pvalue != None:
            if not isinstance(pvalue, float) and not (0 <= pvalue >= 1):
                raise TypeError('El "pvalue" tiene que ser un float entre 0 y 1')
                return None

        if len(columns) == 0:
            numericas = data.select_dtypes(include= np.number).columns
            numericas = numericas.drop(target_col)
        else:
            numericas = data[columns].select_dtypes(include= np.number).columns

        print(numericas)
        print(data.columns)
        lista = []
        for x in numericas:
            datos = data[[x, target_col]].dropna()
            corr, p = pearsonr(datos[x], datos[target_col])
            if abs(corr) < umbral_corr:
                if pvalue is None or p <= (1 - pvalue):
                    lista.append(x)
        data_pairplot = lista + [target_col] # si no es por lista da error (unir diferentes listas)

        for x in range(0, len(lista), 4):
            columnas = lista[x:x+4] + [target_col]
            sns.pairplot(data[columnas])
            plt.show()
        
        return lista
    except Exception as e:
        print(f"Error en la función plot_features_num_regression: {e}")
        return None



def get_features_cat_regression ( data, target_col, p_value = 0.05):
    """
    Escoge las variables categóricas con una relación con una variable numérica.

    Las varaibles pueden ser binarias, no binarias, numérica discreta o continua. En esta función se hacen pruebas estadísticas cómo Chi-2, t-test o ANOVA.

    
    Parámetros
    ----------
    data : DataFrame
        Es el DataFrame con todas las variables contanto la 'target_col'
        
    target_col : str
        Es el nombre de la columna target o valor a predecir

    pvalue : float
        Valor de confianza que por defecto sera de 0.05

    Retorna
    -------
    lista : list
        Devuelve una lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario:
         - t-test: para categóricas no numéricas.
         - chi-2:  para binarias con valores 0/1.
         - ANOVA: para categóricas con valores numéricos.
    
    Excepciones
    -----------
    TypeError
        Si el `data` no es un DataFrame o `target_col` no es una columna numérica válida.
    
    ValueError
        Si `target_col` no está presente en el DataFrame.
     """

    try: 
        if not isinstance (data, pd.DataFrame): # Comprobando que el DataFrame es un DataFrame
            
            raise TypeError ('El argumento "data" tiene que ser un DataFrame')
        
        if not (data[target_col].dtype =='int64' or data[target_col].dtype =='float64'): # Compruebo si es una columna de tipo numérico
           
            raise TypeError ('El argumento introducido como target_col, no es una variable numérica')
        
        if not (data[target_col].nunique() > 10 and (data[target_col].nunique()/len(data)*100 > 25)):# Pongo la cardinalidad > 10 para que coja las numericas discretas y el porcentaje de cardinalida mayor de 25% para cojer las numéricas continuas
            
            raise TypeError ('El argumento introducido como target_col, no es una variable numérica discreta o contínua con alta cardinalidad')
        
        if target_col not in data.columns: # Comprobamos que el target esta en el DataFrame
            raise ValueError('El argumento "target_col" tiene que pertenecer al DataFrame')
        
        # Creamos una dataframe para tipificar cada columna y ver qué variables son categóricas, binarias, numéricas contínuas o discretas
        df_tipificacion = pd.DataFrame([data.nunique(), data.nunique()/len(data) * 100, data.dtypes]).T.rename(columns = {0: "Card",1: "%_Card", 2: "Tipo"})
        
        # Ahora ya si las clasificamos en función de los resultados de las operaciones realizadas en el apartado anterior
        df_tipificacion["Clasificada_como"] = "Categorica" 
        df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
        df_tipificacion.loc[df_tipificacion["Card"] > 10, "Clasificada_como"] ="Numerica Discreta"
        df_tipificacion.loc[df_tipificacion["%_Card"] > 30, "Clasificada_como"] = "Numerica Continua"
        print(df_tipificacion) 

        # Genero un listado para cada una de las clasificaciones anteriores
        # Para categóricas
        categoricas = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Categorica']
        lista_categoricas = categoricas.index.tolist()
        print(lista_categoricas)

        # Para binarias
        binarias = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Binaria']
        lista_binarias = binarias.index.tolist()
        print(lista_binarias)

        # Para numérica discreta
        discretas = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Numerica Discreta']
        lista_discretas = discretas.index.tolist()
        print(lista_discretas)

        # Para numérica contínua
        continuas = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Numerica Continua']
        lista_continuas = continuas.index.tolist()
        print(lista_continuas)

        # Voy a clasificar todas las variables categoricas (categoricas y binarias, según sus valores)
        listado_binarias = [] # Lista de binarias con valores 0-1, True-False
        listado_categoricas = [] # Cajón de sastre donde irán las que no son valores 0-1, True_False, ni numéricas, por tanto string
        listado_categorica_numerica = [] # Lista de categoricas con valores numéricos
        valores = [0,1] # Condición para las binarias con valores 0-1, True-False
        for col in data[lista_binarias].columns: # Para separar las binarias con valores True-False, 0-1
            if data[col].isin(valores).all():
                listado_binarias.append(col)
            else:
                listado_categoricas.append(col) # Resto de categoricas (string)
        
        for col in data[lista_categoricas]: # Para separar las numéricas
            if data[col].dtype == 'int64' or data[col].dtype == 'float64':
                listado_categorica_numerica.append(col)
            else:
                listado_categoricas.append(col) # Resto de categoricas (string)
        
        print(listado_categoricas)
        print(listado_binarias)
        print(listado_categorica_numerica)
        

        listado_completo = [] # Creo lista vacia para ir introduciendo las variables que nos interesan

        # Según el tipo de valor que tenga la variable categórica, irá hacia un test u otro
        # Para las variables categóricas binarias cuyos valores no son 0-1, True-False, puedo hacer un chi-cuadrado. Uso listado_categoricas
        from scipy.stats import chi2_contingency
        for col in data[listado_categoricas].columns:
            tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
            resultado_test = chi2_contingency(tabla)
            if resultado_test[1] < p_value:
                listado_completo.append(col)

        # Prueba t-test con categoricas binarias valores 0-1, True-False. Uso listado_binarias
        import scipy.stats as stats  
        for col in data[listado_binarias].columns:
            x = stats.ttest_ind(data[target_col], data[col].dropna(), alternative='two-sided') #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
            if x[1] < p_value:
                listado_completo.append(col)
                
        # Para las variables categóricas no binarias, pero numéricas tengo que hacer un ANOVA. Uso listado_categorica_numerica 
        for col in data[listado_categorica_numerica].columns:
            resultado_test = stats.f_oneway (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
            if resultado_test[1] < p_value:
                listado_categoricas.append(col)
        return listado_completo
       
    except Exception as e:
        print(f"Error en la función get_feautures-cat_regression: {e}")





def plot_features_cat_regression ( data, target_col = "", columns = [], p_value = 0.05, with_individual_plot = False):
    """
    Escoge las variables categóricas con una relación con una variable numérica.

    Se usan pruebas de hipótesis para obtener la relación estadística entre las variables categóricas y una variable que es el 'target_col', 
    se usan diferentes test estadísticos como:
    Test chi-2, T-test, ANOVA y correlación de Pearsonr

    Generará histogramas, individuales o no individuales

    

    Parámetros
    ----------
    data : DataFrame
        Es el DataFrame con todas las variables contanto la 'target_col'
        
    target_col : str
        Es el nombre de la columna target o valor a predecir

    columns : list
        Lista de columnas, por defecto estará vacía

    pvalue : float
        Valor de confianza que por defecto sera de 0.05
    
    with_individual_plot : bool, default = False
        Si es True, se generará un histograma para cada variable
    

    Retorna
    -------
    dict
        Devuelve un diccionario con las valores de las columnas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario

        
    Excepciones
    -----------
    TypeError
        Si `data` no es un DataFrame
    ValueError
        Si `target_col` no está en el DataFrame o `columns` no es una lista de strings

    """
    

    try: 
        if not isinstance (data, pd.DataFrame): # Comprobando que el DataFrame es un DataFrame
            
            raise TypeError ('El argumento "data" tiene que ser un DataFrame')
    
        if target_col not in data.columns: # Comprobamos que el target esta en el DataFrame
            raise ValueError ('El argumento "target_col" tiene que pertenecer al DataFrame')
        
        if columns != [] and not all(isinstance(elemento, str) for elemento in columns): # Compruebo si es una lista solo de strings
             raise ValueError ('El argumento columns no es una lista de string')
        
        # Después de comprobar que cada argumento es el que tiene que ser...
        # Si columns es una lista de string...
        if columns != []:
            # Comprobamos que valores de la lista de columnas son variables categoricas(binarias + categoricas como tal)
            # Primero generamos un nuevo dataframe con las columnas que nos introduzcan en el listado
            df_seleccionado = data[columns]
            # Creamos una dataframe para tipificar cada columna y ver qué variables son categóricas, binarias, numéricas contínuas o discretas
            df_tipificacion = pd.DataFrame([df_seleccionado.nunique(), df_seleccionado.nunique()/len(df_seleccionado) * 100, df_seleccionado.dtypes]).T.rename(columns = {0: "Card",1: "%_Card", 2: "Tipo"})
        
            # Ahora ya si las clasificamos en función de los resultados de las operaciones realizadas en el apartado anterior
            df_tipificacion["Clasificada_como"] = "Categorica" 
            df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
            df_tipificacion.loc[df_tipificacion["Card"] > 10, "Clasificada_como"] ="Numerica discreta"
            df_tipificacion.loc[df_tipificacion["%_Card"] > 25, "Clasificada_como"] = "Numerica continua"
            print(df_tipificacion) 

            # Genero un listado para cada una de las clasificaciones anteriores
            # Para categóricas
            categoricas = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Categorica']
            lista_categoricas = categoricas.index.tolist()
            print(lista_categoricas)

            # Para binarias
            binarias = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Binaria']
            lista_binarias = binarias.index.tolist()
            print(lista_binarias)

            # Voy a clasificar todas las variables categoricas (categoricas y binarias, según sus valores). Lo usaré en los test si la taget es nmerica continua
            listado_binarias = [] # Lista de binarias con valores 0-1, True-False
            listado_categoricas_string = [] # Cajón de sastre donde irán las que no son valores 0-1, True_False, ni numéricas, por tanto string
            listado_categorica_numerica = [] # Lista de categoricas con valores numéricos
            valores = [0,1] # Condición para las binarias con valores 0-1, True-False
            for col in data[lista_binarias].columns: # Para separar las binarias con valores True-False, 0-1
                if data[col].isin(valores).all():
                    listado_binarias.append(col)
            else:
                listado_categoricas_string.append(col) # Resto de categoricas (string)
        
            for col in data[lista_categoricas]: # Para separar las numéricas
                if data[col].dtype == 'int64' or data[col].dtype == 'float64':
                    listado_categorica_numerica.append(col)
            else:
                listado_categoricas_string.append(col) # Resto de categoricas (string)

            # Hago un listado completo con las columnas categoricas en general
            listado_categoricas = lista_binarias + lista_categoricas
            
            # Ahora que están tipificadas tengo que hacer los test contra la variable target
            # Debo comprobar de qué tipo es la variable target
            clasificacion = 'Categorica'
            if data[target_col].nunique() == 2:
                clasificacion = 'Binaria'
                print ('Binaria')
            elif data[target_col].nunique() > 10 and data[target_col].nunique()/len(data) * 100 < 25:
                clasificacion = 'Numerica discreta'
                print('Numerica Discreta')
            elif data[target_col].nunique() > 10 and data[target_col].nunique()/len(data) * 100 > 25:
                clasificacion = 'Numerica continua'
                print('Numerica continua')
            
            # Tras saber la tipificacion de la variable target, tengo que realizar los test correspondientes y comparar el p-value obtenido con el que tenemos por defecto
            listado_completo = {}
            # Si target y columns son categoricas test chi-cuadrado
            if clasificacion in ('Binaria', 'Categorica'):
                from scipy.stats import chi2_contingency
                for col in data[listado_categoricas].columns:
                    tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    resultado_test = chi2_contingency(tabla)
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])
            # Si target es continua y columns categoricas un ANOVA , t-test (variables binarias) o chi-2
            elif clasificacion == 'Numerica continua':
                # Para categóricas binarias se hace t-test
                import scipy.stats as stats
                for col in data[listado_binarias].columns:
                    resultado_test = stats.ttest_ind(data[target_col], data[col].dropna(), alternative='two-sided') #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])
                # Para las categoricas numéricas se usa ANOVA
                import scipy.stats as stats 
                for col in data[listado_categorica_numerica].columns:
                    resultado_test = stats.f_oneway (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])
                # Para las categoricas string se usa chi-2
                from scipy.stats import chi2_contingency
                for col in data[listado_categoricas_string].columns:
                    tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    resultado_test = chi2_contingency(tabla)
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])
             # Si target es discreta y columns categorica test chi-cuadrado
            if clasificacion == 'Numerica discreta':
                from scipy.stats import chi2_contingency
                for col in data[listado_categoricas].columns:
                    tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    resultado_test = chi2_contingency(tabla)
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])

             # Una vez obtenido el listado de las variables significativas, hay que pintar el histograma
            # Obtengo las claves del diccionario para obtener las columnas para hacer los histogramas            
            columna = listado_completo.keys()
            for col in columna:
                plt.figure (figsize= (len(columna), len(columna)/2))
                sns.histplot(data[col])
            return listado_completo
        # Si columns es una lista vacia... tengo que repetir el proceso, pero con las columnas numéricas
        else:
            # Nuestro Dataframe a usar será el introducido, data
            # Creamos una dataframe para tipificar cada columna y ver qué variables son categóricas, binarias, numéricas contínuas o discretas
            df_tipificacion = pd.DataFrame([data.nunique(), data.nunique()/len(data) * 100, data.dtypes]).T.rename(columns = {0: "Card",1: "%_Card", 2: "Tipo"})
        
            # Ahora ya si las clasificamos en función de los resultados de las operaciones realizadas en el apartado anterior
            df_tipificacion["Clasificada_como"] = "Categorica" 
            df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
            df_tipificacion.loc[df_tipificacion["Card"] > 10, "Clasificada_como"] ="Numerica discreta"
            df_tipificacion.loc[df_tipificacion["%_Card"] > 25, "Clasificada_como"] = "Numerica continua"
            print(df_tipificacion) 

            # Genero un listado para cada una de las clasificaciones anteriores en numericas
            # Para discretas
            discreta = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Numerica discreta']
            lista_discretas = discreta.index.tolist()
            print(lista_discretas)

            # Para continuas
            continuas = df_tipificacion[df_tipificacion['Clasificada_como'] == 'Numerica continua']
            lista_continuas = continuas.index.tolist()
            print(lista_continuas)

            # Genero un listado único de variales numericas
            lista_numericas = (lista_continuas + lista_discretas)
            print(lista_numericas)

            # Ahora que están tipificadas tengo que hacer los test contra la variable target
            # Debo comprobar de qué tipo es la variable target
            clasificacion = 'Categorica'
            if data[target_col].nunique() == 2:
                clasificacion = 'Binaria'
                print ('Binaria')
            elif data[target_col].nunique() > 10 and data[target_col].nunique()/len(data) * 100 < 25:
                clasificacion = 'Numerica discreta'
                print('Numerica Discreta')
            elif data[target_col].nunique() > 10 and data[target_col].nunique()/len(data) * 100 > 25:
                clasificacion = 'Numerica continua'
                print('Numerica continua')

            # Tras saber la tipificacion de la variable target, tengo que realizar los test correspondientes
            listado_completo = {}
            # Si la target es una variable numerica (tanto discreta como contínua), cuyo dtype sea numerico (int o float) hago un test de correlación de pearson,  sino se hará un chi-cuadrado,
            if clasificacion in ('Numerica continua', 'Numerica discreta'):
                # En caso de que mi variable target sea una numérica, debo sacarla del listado de features antes de hacer los test
                lista_numericas.remove(target_col)
                if data[target_col].dtype in (int, float): # Target numerica valores dtype (int o float)
                    from scipy.stats import pearsonr
                    for col in data[lista_numericas].columns:
                        datos = data[[col, target_col]].dropna()
                        resultado_test = pearsonr(datos[col], datos[target_col])
                        if resultado_test[1] < p_value:
                            listado_completo[col] = (1-resultado_test[1])
                else:
                    from scipy.stats import chi2_contingency
                    for col in data[lista_numericas].columns:
                        tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                        resultado_test = chi2_contingency(tabla)
                        if resultado_test[1] < p_value:
                            listado_completo[col] = (1-resultado_test[1])
            # Si la target es una variable binaria con valores 0-1, True-False hago un test de correlación de pearson
            if clasificacion in ('Binaria'):
                if data[target_col].dtype in (int, bool):
                    from scipy.stats import pearsonr
                    for col in data[lista_numericas].columns:
                        datos = data[[col, target_col]].dropna()
                        resultado_test = pearsonr(datos[col], datos[target_col])
                        if resultado_test[1] < p_value:
                            listado_completo[col] = (1-resultado_test[1])
                else: # Cuando la target es binaria pero no tiene valores 0-1, True-False, se hace un test chi-cuadrado
                        from scipy.stats import chi2_contingency
                        for col in data[lista_numericas].columns:
                            tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                            resultado_test = chi2_contingency(tabla)
                            if resultado_test[1] < p_value:
                                listado_completo[col] = (1-resultado_test[1])
            # Si la target es una variable categórica, no numérica (string), se hace un test chi-cuadrado
            if clasificacion == 'Categorica':
                from scipy.stats import chi2_contingency
                for col in data[lista_numericas].columns:
                    tabla = pd.crosstab (data[target_col], data[col].dropna()) #uso el dropna para quitar aquellos valores nulos de la columna y que no me de errores 
                    resultado_test = chi2_contingency(tabla)
                    if resultado_test[1] < p_value:
                        listado_completo[col] = (1-resultado_test[1])

            # Una vez obtenidos los listados de columnas, hay que devolver los histogramas
            columna = listado_completo.keys()
            for col in columna:
                plt.figure (figsize= (len(columna), len(columna)/2))
                sns.histplot(data[col])
            return listado_completo
          

    except Exception as e:
        print(f"Error en la función plot_feautures-cat_regression: {e}")        