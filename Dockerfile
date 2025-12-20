# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalar las dependencias
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc python3-dev
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los scripts de la aplicación y el modelo
COPY app.py .

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5001

# Comando para ejecutar la aplicación cuando se inicie el contenedor
CMD ["python", "app.py"]
