# Utiliser une image de base Python légère et stable
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Empêcher Python d'écrire des fichiers .pyc et activer le mode non-tamponné pour les logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copier et installer les dépendances en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'ensemble du projet
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande par défaut : lancer l'application Streamlit accessible depuis l'extérieur
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

