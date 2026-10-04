# ------------------- Importaciones

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from IPython.display import display
from sklearn.neural_network import MLPClassifier


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

# ------------------- # reegresion logistica

"""primera aproximacion solver newton-cg y C=1.0"""
clf = LogisticRegression(C=1.0, solver="newton-cg", max_iter=1000)
modelo_RL = clf.fit(x_train, y_train)

print("regresion logistica:\naccuracy con el conjunto de validacion = ", modelo_RL.score(x_validation, y_validation))

pr = modelo_RL.predict(x_validation)
print(confusion_matrix(y_validation, pr))

#ajuste de hiperparametros: C
modelo_RL_tmp = LogisticRegression(C=0.2, penalty="l2", solver="lbfgs", max_iter=100, random_state=17)
modelo_RL_tmp.fit(x_train, np.ravel(y_train))
print("C=0.2, lbfgs, l2:", modelo_RL_tmp.score(x_validation, y_validation))

modelo_RL_hyper = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=5, max_iter=5)
modelo_RL_hyper.fit(x_train, np.ravel(y_train))
print("accuracy con max_iter=5 (no converge, por lo tanto, no es confiable): %0.4f" % modelo_RL_hyper.score(x_validation, y_validation))

print("busqueda de C, solver y penalty")
resultados_RL = []
for solver, penalty in [("lbfgs", "l2"), ("newton-cg", "l2"), ("saga", "l2"), ("saga", "l1")]:
    for C in np.logspace(-5, 3, 9):
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=17)
        clf.fit(x_train, np.ravel(y_train))
        tr, va = clf.score(x_train, y_train), clf.score(x_validation, y_validation)
        resultados_RL.append((solver, penalty, C, tr, va))
        print(f"solver={solver}, penalty={penalty}, C={C:g}: train={tr:.4f}, validacion={va:.4f}")

res_RL = pd.DataFrame(resultados_RL, columns=["solver", "penalty", "C", "train", "validacion"])
res_RL = res_RL.sort_values(["validacion", "C"], ascending=[False, True])
display(res_RL.head(5))

mejor_RL = res_RL.iloc[0]
print("mejor combinacion:", mejor_RL["solver"], mejor_RL["penalty"], "C =", mejor_RL["C"])

print("busqueda fina de C alrededor del mejor")
resultados_C = []
for C in np.linspace(mejor_RL["C"] / 2, mejor_RL["C"] * 2, 9):
    clf = LogisticRegression(C=C, penalty=mejor_RL["penalty"], solver=mejor_RL["solver"], max_iter=1000, random_state=17)
    clf.fit(x_train, np.ravel(y_train))
    resultados_C.append((C, clf.score(x_validation, y_validation)))
    print(f"C={C:.4f}: validacion={resultados_C[-1][1]:.4f}")

mejor_C = max(resultados_C, key=lambda t: (t[1], -t[0]))[0]
print("mejor C en validacion:", mejor_C)

"""con C=0.2, lbfgs y l2 la exactitud en validacion es 0.8178 y con max_iter=5 sale 0.8039 pero no converge, por eso ese resultado no se toma en cuenta. en la busqueda, con C muy chico el modelo se regulariza de mas y se queda en 0.58-0.59 que se acerca mas al modelo inicial base, con (C=0.01) sube a 0.81 y desde (C=0.1) se estabiliza en 0.82. los tres solver dan casi lo mismo y l1 con saga tambien. la mejor validacion es 0.8239 (saga con l1 y C=10). la busqueda fina alrededor no cambia nada (0.8233 a 0.8239), el mejor es (C=8.75) train (0.830) y validacion (0.824) quedan muy cerca, asi que no hay sobre-entrenamiento. las diferencias entre combinaciones son de menos de un punto"""

modelo_RL = LogisticRegression(C=mejor_C, penalty=mejor_RL["penalty"], solver=mejor_RL["solver"], max_iter=1000, random_state=17)
modelo_RL.fit(x_train, np.ravel(y_train))

# ------------------- modelo 2: red nueronal (perceptron multicapa)

"""primera aproximacion con dos capas ocultas de 15 y 4 neuronas y max_iter=700, se mide con validacion y se ve su matriz de confusion."""
modelo_NN = MLPClassifier(hidden_layer_sizes=(15, 4), max_iter=700, random_state=42)
modelo_NN.fit(x_train, y_train)
print("red neuronal (15, 4): exactitud validacion =", modelo_NN.score(x_validation, y_validation))
print(confusion_matrix(y_validation, modelo_NN.predict(x_validation)))

"""con la primera aproximacion (15, 4) la red neuronal da 0.8256 en validacion, superior a la del modelo base (0.579) y casi igual que la regresion logistica. ahora se ajustan sus hiperparametros con las curvas de aprendizaje"""

# ------------------- curvas de aprendizaje

neuronas = [i for i in range(1, 50, 5)]
print(neuronas)

"""caso sub-entrenado: alpha muy grande (50), los pesos tienden a cero y el modelo no aprende"""
print("alpha=50")
train_scores, valid_scores = list(), list()
train_errors, valid_errors = list(), list()

for i in neuronas:
    model = MLPClassifier(hidden_layer_sizes=(i, i),
                          max_iter=3000,
                          alpha=50,
                          random_state=42)
    model.fit(x_train, y_train)

    #predicciones y metricas con el conjunto de entrenamiento
    train_yhat = model.predict(x_train)
    train_loss = np.mean(abs(y_train - train_yhat))
    train_errors.append(train_loss)
    train_acc = 1 - train_loss
    train_scores.append(train_acc)

    #predicciones y metricas con el conjunto de validacion
    valid_yhat = model.predict(x_validation)
    valid_loss = np.mean(abs(y_validation - valid_yhat))
    valid_errors.append(valid_loss)
    valid_acc = 1 - valid_loss
    valid_scores.append(valid_acc)

    #evolucion de las metricas durante el entrenamiento
    print("> %d...\ttrainacc: %.3f, validacc: %.3f, trainloss: %.3f, validloss: %.3f"
          % (i, train_acc, valid_acc, train_loss, valid_loss))

plt.plot(neuronas, train_scores, "-o", label="Train")
plt.plot(neuronas, valid_scores, "-o", label="Validacion")
plt.legend()
plt.title("Exactitud: Caso Sub-entrenado / Underfitting")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("exactitud")
plt.show()

plt.plot(neuronas, train_errors, "-o", label="Train")
plt.plot(neuronas, valid_errors, "-o", label="Validacion")
plt.legend()
plt.title("caso sub-entrenado")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("error")
plt.show()

"""con alpha=50 la exactitud queda en 0.579 en entrenamiento y validacion con cualquier cantidad de neuronas. las curvas son planas y estan bajas, es el caso sub-entrenado, los pesos se van a cero y la red no aprende."""

"""caso sobre-entrenado: alpha pequeno (0.15)"""
print("alpha=0.15")
train_scores, valid_scores = list(), list()
train_errors, valid_errors = list(), list()

for i in neuronas:
    model = MLPClassifier(hidden_layer_sizes=(i, i),
                          max_iter=3000,
                          alpha=0.15,
                          random_state=42)

    model.fit(x_train, y_train)

    #predicciones y metricas con el conjunto de entrenamiento
    train_yhat = model.predict(x_train)
    train_loss = np.mean(abs(y_train - train_yhat))
    train_errors.append(train_loss)
    train_acc = 1 - train_loss
    train_scores.append(train_acc)

    #predicciones y metricas con el conjunto de validacions
    valid_yhat = model.predict(x_validation)
    valid_loss = np.mean(abs(y_validation - valid_yhat))
    valid_errors.append(valid_loss)
    valid_acc = 1 - valid_loss
    valid_scores.append(valid_acc)

    #evolucion de las metricas durante el entrenamiento
    print("> %d...\ttrainacc: %.3f, validacc: %.3f, trainloss: %.3f, validloss: %.3f"
          % (i, train_acc, valid_acc, train_loss, valid_loss))

plt.plot(neuronas, train_scores, "-o", label="Train")
plt.plot(neuronas, valid_scores, "-o", label="Validacion")
plt.legend()
plt.title("sobre-entrenado")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("exactitud")
plt.show()

plt.plot(neuronas, train_errors, "-o", label="Train")
plt.plot(neuronas, valid_errors, "-o", label="Validacion")
plt.legend()
plt.title("errores en el caso Sobre-entrenado")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("error")
plt.show()

"""cuando el valor de alpha es 0.15 la exactitud de entrenamiento sube de 0.86 a 0.97 al aumentar las neuronas, pero la de validacion se queda entre 0.83 y 0.84. el modelo memoriza el entrenamiento el caso sobre entrenado"""

"""busqueda del mejor ajuste con alpha intermedio las dos curvas crecen juntas.."""
print("alpha=0.7")
train_scores, valid_scores = list(), list()
train_errors, valid_errors = list(), list()

for i in neuronas:
    model = MLPClassifier(hidden_layer_sizes=(i, i),
                          max_iter=1000,
                          alpha=0.7,
                          random_state=42)
    model.fit(x_train, y_train)

    #predicciones y metricas con el conjunto de entrenamiento
    train_yhat = model.predict(x_train)
    train_loss = np.mean(abs(y_train - train_yhat))
    train_errors.append(train_loss)
    train_acc = 1 - train_loss
    train_scores.append(train_acc)

    #predicciones y metricas con el conjunto de validacion
    valid_yhat = model.predict(x_validation)
    valid_loss = np.mean(abs(y_validation - valid_yhat))
    valid_errors.append(valid_loss)
    valid_acc = 1 - valid_loss
    valid_scores.append(valid_acc)

    #evolucion de las metricas durante el entrenamiento
    print("> %d...\ttrainacc: %.3f, validacc: %.3f, trainloss: %.3f, validloss: %.3f"
          % (i, train_acc, valid_acc, train_loss, valid_loss))

plt.plot(neuronas, train_scores, "-o", label="Train")
plt.plot(neuronas, valid_scores, "-o", label="Validacion")
plt.legend()
plt.title("exactitud caso del mejor ajuste")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("exactitud")
plt.show()

plt.plot(neuronas, train_errors, "-o", label="Train")
plt.plot(neuronas, valid_errors, "-o", label="Validacion")
plt.legend()
plt.title("errores en el caso del mejor ajuste")
plt.xlabel("neuronas en cada capa, de 2 ocultas")
plt.ylabel("error")
plt.show()

print("busqueda de alpha y neuronas")
resultados_NN = []
for alpha in [0.7, 1.0, 3.0, 10.0]:
    for i in neuronas:
        model = MLPClassifier(hidden_layer_sizes=(i, i), max_iter=1000, alpha=alpha, random_state=42)
        model.fit(x_train, y_train)
        tr, va = model.score(x_train, y_train), model.score(x_validation, y_validation)
        resultados_NN.append((alpha, i, tr, va))
        print(f"alpha={alpha}, neuronas={i}: train={tr:.3f}, validacion={va:.3f}")

res_NN = pd.DataFrame(resultados_NN, columns=["alpha", "neuronas", "train", "validacion"])
display(res_NN.sort_values("validacion", ascending=False).head(5))

mejor = res_NN.sort_values("validacion", ascending=False).iloc[0]
mejor_alpha, mejor_neuronas = mejor["alpha"], int(mejor["neuronas"])
print("mejor alpha:", mejor_alpha, "- mejor neuronas por capa:", mejor_neuronas)

"""cuando tengo que (alpha=0.7) las curvas de entrenamiento y validacion crecen cerca una de la otra por eso es el mejor ajuste. alpha mas grande (entre 3-10) vuelve a bajar la validacion a 0.82 y 0.80 por que empieza el sub-entrenamiento, entonces el mejor en validacion es (alpha=0.7) con 21 neuronas por capa (0.8511) el cual empata con (alpha=1.0) con 11 neuronas, la exactitud de entrenamiento baja respecto al sobre-entrenado y eso es normal."""

modelo_NN = MLPClassifier(hidden_layer_sizes=(mejor_neuronas, mejor_neuronas),
                          max_iter=1000, alpha=mejor_alpha, random_state=42)
modelo_NN.fit(x_train, y_train)

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
