import joblib
from flask import Flask, request, jsonify
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient(tracking_uri=tracking_uri)
latest_version_info = client.get_latest_versions("iris-rf")[0]
model_uri = f"models:/iris-rf/{latest_version_info.version}"
model = mlflow.sklearn.load_model(model_uri)


# Inicializar la aplicación Flask
app = Flask(__name__)


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
        # Devolver la predicción en formato JSON
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
