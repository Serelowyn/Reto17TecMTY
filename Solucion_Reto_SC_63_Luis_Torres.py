# ------------------- Importaciones

import pandas as pd

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
# ------------------- 
# ------------------- 
# ------------------- 
# ------------------- 
