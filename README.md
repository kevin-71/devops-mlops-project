# Climate ML MLOps Project

Application MLOps pour comparer plusieurs modèles de régression sur l'évolution conjointe des anomalies de température globale et du CO2 atmosphérique.

Le projet suit une architecture simple mais complète :

- un backend FastAPI qui prépare les données, entraîne les modèles et sert les prévisions ;
- un frontend Streamlit qui consomme l'API ou le service local ;
- des jeux de données bruts dans `data/` ;
- des artefacts modèles persistés dans `models/` ;
- le versionnement des jeux de données et des modèles avec DVC ;
- le suivi des expériences (métriques, paramètres, modèles) avec MLflow, hébergé sur DagsHub ;
- une CI GitHub Actions pour exécuter les tests ;
- une orchestration Docker Compose pour lancer backend et frontend ensemble.

## Architecture globale

```mermaid
flowchart LR
    A[Data sources\n/data] --> B[backend/climate/data.py\nnettoyage et fusion]
    B --> C[backend/climate/features.py\nsplit temporel]
    C --> D[backend/climate/models.py\nentraînement et métriques]
    D --> E[backend/climate/service.py\npersistance et prévision]
    E --> F[backend/main.py\nAPI FastAPI]
    F --> G[frontend/app.py\nDashboard Streamlit]
    E --> H[models/\njoblib, keras, scaler]
    I[tests/test_data.py] --> B
```

Flux de données principal :

1. Les CSV sont lus depuis `data/`.
2. `backend/climate/data.py` fusionne température et CO2, puis crée un frame exploitable.
3. `backend/climate/features.py` construit le split temporel et les variables d'entrée.
4. `backend/climate/models.py` entraîne les modèles classiques et, si disponible, les modèles deep learning.
5. `backend/climate/service.py` stocke l'état, charge les artefacts et génère les prévisions futures.
6. `backend/main.py` expose ces capacités via HTTP.
7. `frontend/app.py` affiche les résultats, les métriques et la projection.

---

# Structure du projet

## Racine

- `README.md` : documentation d'architecture, d'exécution et de maintenance.
- `docker-compose.yml` : orchestration des services backend et frontend.
- `Dockerfile.backend` : image Docker du backend FastAPI.
- `Dockerfile.frontend` : image Docker du frontend Streamlit.
- `requirements.txt` : dépendances du frontend et des usages légers.
- `requirements-backend.txt` : dépendances complètes du backend.
- `requirements-dev.txt` : dépendances pour les outils de qualité de code.
- `.env.example` : modèle des variables d'environnement nécessaires (DagsHub/MLflow). À copier en `.env` (jamais commité) avec les vraies valeurs.
- `load-env.ps1` : script PowerShell pour charger les variables du `.env` dans la session courante (nécessaire sous Windows avant `python -m backend.train`). Si l'exécution est bloquée par la politique PowerShell, lancer d'abord `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` dans la même session.
- `.pre-commit-config.yaml` : configuration des hooks pre-commit.
- `.dvc/` : configuration de DVC et du stockage distant.
- `data.dvc` : suivi de version du dossier `data`.
- `models.dvc` : suivi de version du dossier `models`.
- `models/` : répertoire de persistance des artefacts entraînés.
- `data/` : jeu de données principal utilisé par le code.
- `tests/` : tests automatisés.
- `.github/workflows/` : pipeline GitHub Actions.
- `.gitignore` : exclusions Git.

## Backend

- `backend/__init__.py` : marque le package Python `backend`.
- `backend/main.py` : point d'entrée FastAPI. Déclare les routes `/health`, `/status`, `/train`, `/metrics` et `/forecast`.
- `backend/train.py` : script CLI pour entraîner ou ré-entraîner les modèles depuis le terminal.
- `backend/climate/__init__.py` : exports publics du sous-module météo.
- `backend/climate/settings.py` : définit les chemins projet (`data/`, `models/`), crée les dossiers nécessaires, et lit la configuration MLflow/DagsHub depuis les variables d'environnement.
- `backend/climate/data.py` : charge les CSV, nettoie les colonnes, fusionne température et CO2, construit le dataset principal et les lags.
- `backend/climate/features.py` : définit les colonnes d'entrée et le découpage temporel train/test.
- `backend/climate/models.py` : construit et entraîne les modèles classiques et optionnels deep learning, puis calcule RMSE, MAE et R².
- `backend/climate/forecasting.py` : génère les dates futures, projette le CO2 et produit les prévisions de température.
- `backend/climate/service.py` : orchestre tout le pipeline, persiste l'état, charge les artefacts, expose les métriques, fabrique les réponses de prévision, et journalise chaque entraînement dans MLflow si configuré.

## Frontend

- `frontend/app.py` : interface Streamlit. Affiche les métriques, les graphiques historiques et la prévision future. Peut communiquer avec le backend via `BACKEND_URL` ou utiliser directement le service local.

## Données

- `data/GLB.Ts+dSST.csv` : source des anomalies de température globale.
- `data/co2_mm_mlo.csv` : source des mesures de CO2 atmosphérique.

## Artefacts

- `models/climate_artifacts.joblib` : état sérialisé du pipeline après entraînement.
- `models/climate_summary.json` : résumé lisible de l'entraînement.
- `models/*.keras` : modèles deep learning sauvegardés.
- `models/dl_scaler.joblib` : scaler utilisé pour ANN/CNN/GCN.
- `models/.gitkeep` : conserve le dossier dans Git lorsqu'il est vide.

## Tests et CI

- `tests/test_data.py` : vérifie que les frames de données sont construites correctement et contiennent les colonnes attendues.
- `.github/workflows/ci.yml` : installe les dépendances backend puis exécute automatiquement les tests avec `pytest`. Le job est déjà prêt à recevoir un secret `DAGSHUB_USER_TOKEN` pour une future intégration MLflow en CI (non utilisée pour l'instant, voir section dédiée plus bas).

---

# Comment chaque couche communique

- Le frontend appelle d'abord l'API backend si `BACKEND_URL` est défini.
- Sinon, Streamlit instancie directement `ClimateService` pour un mode local.
- `ClimateService` charge ou entraîne les modèles, puis persiste les artefacts dans `models/`.
- Les prévisions s'appuient sur l'historique, les lags temporels et une projection du CO2.
- Les jeux de données et les modèles sont versionnés avec DVC afin de garantir la reproductibilité des entraînements.

---

# Lancer en local

1. Installer les dépendances frontend.

```bash
pip install -r requirements.txt
```

2. Installer les dépendances backend.

```bash
pip install -r requirements-backend.txt
```

3. Installer les outils de développement (optionnel).

```bash
pip install -r requirements-dev.txt
```

4. Télécharger les données et modèles versionnés avec DVC.

```bash
dvc pull
```

> ⚠️ Toujours faire `dvc pull` (après un `git pull`) avant de lancer un entraînement ou de démarrer l'application, afin de travailler sur les dernières données et les derniers modèles versionnés. Sans ce pull, `data/` et `models/` peuvent contenir une version locale obsolète ou incomplète.

5. Démarrer l'API backend.

```bash
uvicorn backend.main:app --reload --port 8000
```

6. Démarrer Streamlit.

```bash
streamlit run frontend/app.py
```

---

# Versionnement des données

Le projet utilise **DVC (Data Version Control)** afin de gérer les jeux de données et les modèles indépendamment du dépôt Git.

Les dossiers suivis par DVC sont :

- `data/`
- `models/`

Les fichiers `data.dvc` et `models.dvc` permettent de retrouver précisément la version des données utilisée lors d'un entraînement.

Les données sont stockées sur un stockage distant DVC tandis que Git ne versionne que les métadonnées (`*.dvc`).

## Configuration du remote DagsHub (par personne)

Le remote `dagshub` est déjà déclaré dans `.dvc/config` (fichier commité, ne contient que l'URL). Chaque membre de l'équipe doit en revanche configurer **ses propres identifiants** dans `.dvc/config.local` (fichier non commité, ignoré par Git) :

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <votre-nom-utilisateur-dagshub>
dvc remote modify dagshub --local password <votre-token-dagshub>
```

Le `<votre-token-dagshub>` est le même token que celui utilisé pour `DAGSHUB_USER_TOKEN` dans `.env` (voir section MLflow ci-dessous) — c'est le même token DagsHub, mais il doit être renseigné séparément à deux endroits différents (`.dvc/config.local` pour DVC, `.env` pour MLflow), car les deux outils ne partagent aucune configuration entre eux.

**Ne mettez jamais votre token dans `.dvc/config`** (celui commité) : `user`, `auth` et `password` doivent uniquement vivre dans `.dvc/config.local`.

## Récupérer les dernières données

Avant toute session de travail (entraînement, exploration, lancement de l'application), toujours synchroniser code et données dans cet ordre :

```bash
git pull
dvc pull
```

`git pull` récupère les derniers pointeurs `data.dvc` / `models.dvc`, et `dvc pull` télécharge le contenu réel correspondant depuis le remote DagsHub. Sauter cette étape peut faire travailler sur une version de données périmée, incomplète, voire vide.

## Mise à jour des données

Après modification des données :

```bash
dvc add data
git add data.dvc
git commit -m "Update dataset"
dvc push
git push
```

Les autres membres de l'équipe récupèrent ensuite les nouvelles données avec :

```bash
git pull
dvc pull
```

Chaque version des données est ainsi associée à un commit Git, ce qui garantit la reproductibilité des entraînements.

---

# Suivi des expériences avec MLflow (DagsHub)

Le projet peut journaliser chaque entraînement (paramètres, métriques RMSE/MAE/R², modèles) dans **MLflow**, hébergé gratuitement par **DagsHub**. Ce suivi est optionnel : si aucune configuration n'est fournie, l'entraînement fonctionne exactement comme avant, sans aucun appel à MLflow.

## Configuration

1. Récupérer un token DagsHub : sur [dagshub.com](https://dagshub.com), avatar (haut à droite) → **Settings** → **Tokens**. Le **Default Access Token** convient très bien pour un usage personnel.
2. Copier `.env.example` en `.env` à la racine du projet, puis renseigner :

```
DAGSHUB_USER_TOKEN=<votre-token>
DAGSHUB_REPO_OWNER=kevin-71
DAGSHUB_REPO_NAME=climate-mlops
MLFLOW_EXPERIMENT_NAME=climate-ml
```

3. **Ne jamais commiter `.env`** (il doit rester dans `.gitignore`). Chaque membre de l'équipe génère et utilise **son propre token**, afin que les runs MLflow soient correctement attribués à leur auteur.

4. Charger les variables d'environnement avant de lancer l'entraînement. Windows/PowerShell ne lit pas automatiquement un fichier `.env`, contrairement à `export $(cat .env | xargs)` sous Linux/macOS. Un script `load-env.ps1` est fourni à la racine du projet pour ça :

```powershell
.\load-env.ps1
```

Si PowerShell bloque l'exécution du script (politique d'exécution non signée), lancer d'abord dans la même session :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\load-env.ps1
```

Sous Linux/macOS, l'équivalent est :

```bash
export $(grep -v '^#' .env | xargs)
```

## Lancer un entraînement journalisé

Avant de lancer l'entraînement, s'assurer d'avoir les dernières données :

```bash
git pull
dvc pull
```

Puis lancer l'entraînement :

```bash
python -m backend.train --refresh
```

Chaque appel à `ClimateService.train()` ouvre un run parent MLflow (`climate-training`), puis un run imbriqué par modèle (Linear Regression, Decision Tree, Random Forest, XGBoost, ANN, CNN, GCN) avec ses hyperparamètres, ses métriques et son artefact modèle. Les résultats sont visibles dans l'onglet **Experiments** du dépôt DagsHub.

## CI (à venir)

Le pipeline `.github/workflows/ci.yml` est déjà prêt à recevoir la configuration DagsHub via un secret GitHub nommé `DAGSHUB_USER_TOKEN` (Settings du dépôt → Secrets and variables → Actions), mais la CI n'exécute pour l'instant que les tests (`pytest`) et ne lance pas d'entraînement complet. Cette intégration sera activée plus tard si l'on souhaite journaliser automatiquement des runs à chaque push.

---

# Qualité de code

Le projet automatise le formatage et le linting grâce à **pre-commit**.

Installation :

```bash
pip install -r requirements-dev.txt
pre-commit install
```

À chaque `git commit`, les hooks exécutent automatiquement les vérifications et les corrections nécessaires afin de maintenir une qualité de code homogène.

---

# Collaboration

Le code source est versionné avec Git tandis que les données et les modèles sont versionnés avec DVC.

Workflow recommandé :

```text
git clone
        │
        ▼
pip install -r requirements-backend.txt
        │
        ▼
dvc pull
        │
        ▼
Lancement de l'application
```

Lorsqu'un membre de l'équipe met à jour les données :

```bash
dvc add data
git add data.dvc
git commit -m "Update dataset"
dvc push
git push
```

Les autres développeurs synchronisent ensuite leur environnement avec :

```bash
git pull
dvc pull
```

> À faire systématiquement en début de session de travail, même si aucune donnée n'a été mise à jour récemment : c'est le seul moyen de garantir que `data/` et `models/` en local correspondent bien à la dernière version partagée par l'équipe.

---

# Avec Docker Compose

```bash
docker compose up --build
```

Le backend est accessible sur :

```
http://localhost:8000
```

Le frontend Streamlit est accessible sur :

```
http://localhost:8501
```

---

# Données

Les fichiers sources sont détectés automatiquement dans `data/`, versionnés avec DVC. Toujours exécuter `dvc pull` (après `git pull`) pour s'assurer de disposer de la dernière version avant de travailler dessus.