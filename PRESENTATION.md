# Script de présentation (démo orale)

Durée totale estimée : 8–12 minutes

Objectif : présenter rapidement l'architecture, montrer l'entraînement d'un candidat, exécuter les quality gates, et expliquer la promotion en production.

1) Introduction (30s)
- Saluer le professeur, dire votre nom et le rôle du projet : "Démo d'une pipeline MLOps pour un modèle de prévision climatique."
- Résumer en une phrase : données versionnées (DVC), entraînement & évaluation, quality gates, promotion (MLflow ou fallback local).

2) Architecture & fichiers clés (1m)
- Montrer le repo (liste rapide) et expliquer chaque composant :
  - `backend/` : API FastAPI, entraînement (`train.py`) et service métier (`climate/service.py`).
  - `scripts/quality_gates.py` : exécute RMSE / latence et promeut le modèle.
  - `docker-compose.yml` : orchestre `backend`, `frontend`, `pipeline` (pipeline exécute training + gates).
  - `data/` (DVC), `models/` (artefacts locaux), `.github/workflows/` (CI & promotion).

3) Préparer la démo (30s)
- Expliquer les prérequis : Docker + Docker Compose ou Python + dépendances.
- Variables d'environnement optionnelles : `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (si vous voulez montrer MLflow UI/DagsHub).

4) Commandes à exécuter (live) — Cas Docker (recommandé)

Ouvrir un terminal et lancer :

```bash
docker compose up -d --build
```

Explication : cela construit les images et démarre les services. Le service `pipeline` exécute l'entraînement et les quality gates.

Suivre les logs du pipeline :

```bash
docker compose logs -f pipeline
```

Points à montrer dans les logs :
- Début d'entraînement : marqueur `=== TRAIN START ===` et sortie des métriques candidates
- Résumé écrit dans `models/climate_summary.json`
- Exécution des quality gates (RMSE, latence, smoke tests) et message PASS/FAIL
- Si promotion locale : création de `models/production.json`

5) Commandes à exécuter (sans Docker)

Si vous préférez exécuter en local Python :

```bash
pip install -r requirements-backend.txt
python -m backend.train --refresh
python scripts/quality_gates.py
```

Montrez ensuite :
- `cat models/climate_summary.json` — résumé des métriques et modèle candidat
- `cat models/production.json` — si la promotion a eu lieu (fallback local)

6) Tester l'API (optionnel, 1m)

Dans un autre terminal, vérifier la santé et la latence :

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/forecast?horizon=1
```

Montrer la réponse JSON rapide et expliquer le endpoint `GET /forecast` qui sert pour les smoke tests.

7) Montrer MLflow (optionnel, 1–2m)
- Si `MLFLOW_TRACKING_URI` est configuré et accessible, ouvrir l'UI MLflow (adresse selon `MLFLOW_TRACKING_URI`) et montrer :
  - runs enregistrés pour l'entraînement
  - Model Registry (`Climate_Model`) et versions
  - si promotion automatique : la version en `Production`

8) Points techniques à souligner (1m)
- Quality gates : assurent qu'un modèle ne monte en production que si seuils RMSE & latence respectés.
- Fallback local : le pipeline fonctionne sans serveur MLflow (utile en local/CI).
- CI : `.github/workflows/promote.yml` exécute la même logique sur pushes vers `staging`.
- DVC : les données sont versionnées — CI fait un `dvc pull` best-effort. Expliquez la stratégie de tests quand les données manquent.

9) Scénario de démonstration rapide (2–3m) — script oral
- (0:00) "Je lance le pipeline qui entraîne le modèle candidat et exécute les gates."
- (0:10) Lancer `docker compose up -d --build` (ou `python -m backend.train --refresh`).
- (0:40) Ouvrir `docker compose logs -f pipeline` et commenter les étapes : entraînement, métriques, summary.
- (1:20) Montrer le résultat des quality gates : "Gate RMSE : 0.XX (seuil 0.YY) — PASS" ou "FAIL".
- (1:45) Si PASS : montrer `models/production.json` et/ou MLflow Model Registry.
- (2:00) Appeler le endpoint `GET /forecast` pour prouver que le service répond correctement.
- (2:20) Conclusion rapide : robustesse via gates, traçabilité via MLflow/DVC, CI intégrée.

10) Questions & pistes d'amélioration
- Automatiser le rollback si la production baisse après déploiement
- Ajouter plus de gates (schéma de features, tests d'intégration end-to-end, tests d'éthiques/biased)
- Intégrer métriques d'observabilité (Prometheus/Grafana déjà présents dans compose)

Fichiers à montrer au professeur (liste rapide)
- `README.md` — architecture et instructions
- `docker-compose.yml` — orchestration
- `backend/train.py` et `backend/climate/service.py` — logique d'entraînement & enregistrement
- `scripts/quality_gates.py` — conditions de promotion
- `models/climate_summary.json`, `models/production.json` — artefacts produits

Fin — demandez si le prof veut une démo plus longue (ex: mode MLflow complet, ajout de GPU, ou tests CI complets avec DVC remote).
