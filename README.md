# Climate ML MLOps Project

Project Streamlit pour comparer des modèles de régression sur les anomalies de température globale et le CO2 atmosphérique.

Le projet inclut des modèles classiques (régression linéaire, arbre de décision, random forest, XGBoost) et des modèles deep learning (ANN, CNN, GCN).
La projection future est configurée pour aller jusqu'à 25 ans, soit 300 mois.

## Structure

- `backend/` : logique de chargement des données, entraînement et API FastAPI.
- `frontend/` : application Streamlit.
- `data/` : jeux de données bruts.
- `models/` : artefacts entraînés.
- `tests/` : tests de base.
- `.github/workflows/` : CI GitHub Actions.

## Lancer en local

1. Installer les dépendances.
   ```bash
   pip install -r requirements.txt
   ```
2. Installer les dépendances backend (API + entraînement ANN/CNN/GCN).
   ```bash
   pip install -r requirements-backend.txt
   ```
3. Démarrer l'API backend.
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
4. Démarrer Streamlit.
   ```bash
   streamlit run frontend/app.py
   ```

## Avec Docker Compose

```bash
docker compose up --build
```

Le service backend expose l'API sur `http://localhost:8000` et Streamlit sur `http://localhost:8501`.

## Données

Les fichiers sources sont détectés automatiquement dans `data/`. Si besoin, le code sait aussi lire l'ancien dossier `code/`.
