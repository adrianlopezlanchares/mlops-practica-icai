import joblib
from flask import Flask, request, jsonify, Response
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

PREDICTION_COUNTER = Counter(
    "iris_prediction_count",
    "Contador de predicciones del modelo Iris por especie",
    ["species"],
)

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient(tracking_uri=tracking_uri)
latest_version_info = client.get_latest_versions("iris-rf")[0]
model_uri = f"models:/iris-rf/{latest_version_info.version}"
model = mlflow.sklearn.load_model(model_uri)


# Inicializar la aplicación Flask
app = Flask(__name__)


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(
            {"error": "Modelo no cargado. Por favor, entrene el modelo primero."}
        ), 500
    try:
        # Obtener los datos de la petición en formato JSON
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        # Realizar la predicción
        prediction = model.predict(features)
        prediction_int = int(prediction[0])

        # Mapear el resultado numérico a una especie
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map.get(prediction_int, "unknown")

        PREDICTION_COUNTER.labels(species=predicted_species).inc()

        return jsonify({"prediction": prediction_int, "species": predicted_species})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
