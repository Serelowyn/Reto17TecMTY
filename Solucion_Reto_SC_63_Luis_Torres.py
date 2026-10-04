# ------------------- Importaciones

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------- Fin de las importaciones

df = pd.read_csv(r"bank_marketing_RETO_DS_AS.csv")

"""primero para revisar la integridad del archivo"""
print(df.head(10))
print(df.info())
print(df.dtypes)
print(df.shape)
#datos analiticos
print(df.describe())
#verificacion de celdas vacias, aunque en este caso se use la palabra "unkwown" para los datos que no se saben
print(df.isnull().sum())

#para los unkwown
print("unknown")
for columna in df.columns[df.dtypes == object]:
    n_unknown = (df[columna] == "unknown").sum()
    print(f"{columna}: {n_unknown}")

"""los datos perdidos estan en job = 51, education = 377, contact = 1982 y poutcome = 6783. no se hace dropna porque se perderia poutcome. unknown en poutcome significa que el cliente no fue contactado antes por lo cual nos sirve como una variable categorica"""

print(pd.crosstab(df["poutcome"], df["pdays"] == -1))
#hay 6781 unknowns en poutcome

# ------------------- tipos de variables

"""las variables con menos de 20 valores distintos se consideran categoricas"""
print(df.nunique())

num_cols = [c for c in df.columns if df[c].dtype != object and df[c].nunique() >= 20]
cat_cols = [c for c in df.columns if c not in num_cols]

print("numericas:", len(num_cols), num_cols)
print("categoricas:", len(cat_cols), cat_cols)

# ------------------- variable de salida

print(df["y"].value_counts())
print("No: %.1f" % (100 * sum(df["y"] == "no") / df.shape[0]))
print("Si: %.1f" % (100 * sum(df["y"] == "yes") / df.shape[0]))

"""5213 clientes no adquirieron el plan y 3787 si. se tiene 57.9% de exactitud, eso me quiere decir que todos los modelos tienen que superar ese porcentaje."""

modelo_base = sum(df["y"] == "no") / df.shape[0]
print(f"modelo base: {modelo_base:.3f}")

# ------------------- histograma para ver la distribucion

df[num_cols].hist(bins=30, figsize=(14, 8))
plt.tight_layout()
plt.show()

"""balance, duration, campaign, pdays y previous tienen sesgo positivo, no son acampanadas como pide la regresion logistica"""

# -------------------# transformacion de datos categoricos y numericos
#copia del df para el modelo
df_model = df.copy()

"""para el sesgo positivo se aplica una transformacion logaritmica. se usa log(1 + x) en duration, campaign y previous porque son no negativas y tienen ceros"""
sesgadas = ["duration", "campaign", "previous"]
for col in sesgadas:
    df_model[col] = np.log1p(df_model[col])

df_model[sesgadas].hist(bins=30, figsize=(12, 3), layout=(1, 3))
plt.tight_layout()
plt.show()

"""despues del logaritmo duration y campaign se ven mejores, mas simetricas, a excepcion de previous que sigue teniendo muchos 0."""

"""las binarias (yes/no): default, housing, loan y (y) se pasan a 1:yes/0:no. la clase 1 ha adquirio el plan."""
binarias = ["default", "housing", "loan", "y"]
for columna in binarias:
    df_model[columna] = (df_model[columna] == "yes").astype(int)

"""las categoricas con mas de dos niveles se pasan a OHEncoder. drop_first=True sirve para evitar repetir la info"""
multiclase = ["job", "marital", "education", "contact", "month", "poutcome"]
df_model = pd.get_dummies(df_model, columns=multiclase, drop_first=True, dtype=int)

#revisar
print(df_model.shape)
print(df_model.head())
"""7 numericas, 3 binarias y 32 dummies mas la variable de salida."""

X = df_model.drop(columns="y")
y = df_model["y"]

# ------------------- Particion en entrenamiento, validacion y prueba

"""se elige 60 entrenamiento, 20 validacion y 20 prueba, con train_test_split como en las lecciones. stratify mantiene la misma proporcion de (si/no) en los tres conjuntos y los pesos se calculan con entrenamiento, los hiperparametros se escogen con validacion y prueba solo se usa al final."""
x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=11, stratify=y)
x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=11, stratify=y_temp)

print(x_train.shape, x_validation.shape, x_test.shape)
print(y_train.mean(), y_validation.mean(), y_test.mean())

# ------------------- # estandarizacion -> (x - media) / desviacion

def fun2(X, media, desv):
    return (X - media) / desv

"""la media y la desviacion se calculan solo con entrenamiento y se aplican igual a validacion y prueba"""
media = x_train[num_cols].mean()
desv = x_train[num_cols].std()

x_train = x_train.copy()
x_validation = x_validation.copy()
x_test = x_test.copy()
x_train[num_cols] = fun2(x_train[num_cols], media, desv)
x_validation[num_cols] = fun2(x_validation[num_cols], media, desv)
x_test[num_cols] = fun2(x_test[num_cols], media, desv)

print(x_train[num_cols].describe().round(2))

"""las variables numericas de entrenamiento quedan con media 0 y desviacion std 1 lo que me indica un buen rango."""

# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
