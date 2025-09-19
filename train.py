import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

import dagshub
dagshub.init(repo_owner='adrianlopezlanchares', repo_name='mlops-practica-icai', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

# Cargar el conjunto de datos
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Inicializar y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(model, 'model.pkl')

print("Modelo entrenado y guardado como 'model.pkl'")
