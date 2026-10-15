# ML Churn Project — Industrialisation d'un modèle de prédiction du churn

Ce projet transforme un modèle de Machine Learning de prédiction du churn (désabonnement des clients d'un opérateur télécom) en **application industrielle** : déployable, testée, monitorée et maintenable.

Dépôt : https://github.com/christianabakar/ml-churn-project

---

## Sommaire

1. [Architecture](#1-architecture)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Stack technique](#3-stack-technique)
4. [Guide d'installation](#4-guide-dinstallation)
5. [Utilisation de l'API](#5-utilisation-de-lapi)
6. [Tests](#6-tests)
7. [Intégration continue (GitHub Actions)](#7-intégration-continue-github-actions)
8. [Logs centralisés (ELK)](#8-logs-centralisés-elk)
9. [Monitoring (Prometheus + Grafana)](#9-monitoring-prometheus--grafana)
10. [Déploiement sur un serveur](#10-déploiement-sur-un-serveur)
11. [Corrections apportées au TP](#11-corrections-apportées-au-tp)
12. [Résultats des tests de charge](#12-résultats-des-tests-de-charge)
13. [Limites et pistes d'amélioration](#13-limites-et-pistes-damélioration)
14. [Dépannage](#14-dépannage)

---

## 1. Architecture

```
                    ┌──────────────┐
  Client / JMeter ─►│  API FastAPI │──► models/model.pkl (Random Forest)
  Selenium (Swagger)│   :8000      │
                    └──┬────────┬──┘
                       │        │
          logs/api.log │        │ /metrics
                       ▼        ▼
                 ┌──────────┐ ┌────────────┐     ┌──────────┐
                 │ Logstash │ │ Prometheus │────►│ Grafana  │
                 └────┬─────┘ │   :9090    │     │  :3000   │
                      ▼       └────────────┘     └──────────┘
              ┌───────────────┐
              │ Elasticsearch │◄── Kibana :5601
              │     :9200     │
              └───────────────┘

  GitHub (main / develop / feature/*) ──► GitHub Actions (CI : pytest)
```

Chaîne industrielle : **Données → Entraînement → Modèle → API → Tests → Docker → CI → Déploiement → Logs → Monitoring**.

---

## 2. Structure du dépôt

```
ml-churn-project/
├── api/main.py                     # API FastAPI (prédiction, logs, métriques)
├── data/raw/churn.csv              # Dataset généré (2000 clients)
├── notebooks/exploration.py        # Exploration des données
├── src/
│   ├── generate_data.py            # Génération du dataset
│   └── train.py                    # Entraînement et comparaison des modèles
├── models/
│   ├── model.pkl                   # Modèle entraîné (Random Forest)
│   └── README.md                   # Fiche du modèle
├── tests/
│   ├── test_api.py                 # Tests Pytest (4 tests)
│   ├── test_selenium.py            # Test E2E Selenium (Swagger UI)
│   └── jmeter/churn_load_test.jmx  # Plan de test de charge JMeter
├── docker/logstash/logstash.conf   # Pipeline Logstash
├── monitoring/
│   ├── prometheus/prometheus.yml   # Configuration Prometheus
│   └── grafana/ml-churn-dashboard.json  # Dashboard Grafana exporté
├── .github/workflows/ci.yml        # Pipeline CI
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 3. Stack technique

| Domaine | Outil | Version |
|---|---|---|
| Langage | Python | 3.12 |
| ML | scikit-learn, pandas, numpy, joblib | voir `requirements.txt` |
| API | FastAPI, Uvicorn, Pydantic | — |
| Tests | Pytest, Selenium, Apache JMeter | JMeter 5.6.3 |
| Conteneurs | Docker, Docker Compose | — |
| CI | GitHub Actions | — |
| Logs | Elasticsearch, Logstash, Kibana | 8.17.0 |
| Monitoring | Prometheus, Grafana | Prometheus v2.55.1 |

### Le modèle

- **Dataset** : 2000 clients synthétiques. Variables : `age`, `anciennete` (mois), `revenu`, `nb_reclamations`, `montant_mensuel`, `retards_paiement`. Cible : `churn` (0/1).
- **Modèles comparés** (F1-score sur le jeu de test) : Logistic Regression 0,510, Decision Tree 0,545, Random Forest 0,487.
- **Modèle déployé** : Random Forest (200 arbres), conformément au TP. Version `1.0.0`.

---

## 4. Guide d'installation

### 4.1 Prérequis

- Ubuntu 22.04 / 24.04 (natif ou WSL2 sous Windows 11)
- Python 3.12, Git
- Docker et Docker Compose v2 (sous Windows : Docker Desktop avec l'intégration WSL activée)
- 8 Go de RAM minimum pour la stack complète (ELK est gourmand), 16 Go recommandés

Vérification :

```bash
lsb_release -a
python3 --version
git --version
docker --version
docker compose version
```

Si Docker refuse l'accès (`permission denied ... docker.sock`) :

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 4.2 Récupérer le projet

```bash
git clone https://github.com/christianabakar/ml-churn-project.git
cd ml-churn-project
```

### 4.3 Option A — Exécution locale (développement)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Ré)générer les données et le modèle si besoin
python src/generate_data.py
python src/train.py

# Lancer l'API
uvicorn api.main:app --reload
```

API disponible sur http://127.0.0.1:8000 et documentation sur http://127.0.0.1:8000/docs.

### 4.4 Option B — Stack complète avec Docker Compose (recommandé)

```bash
docker compose up -d --build
docker compose ps
```

Les 5 services doivent être à l'état `Up` :

| Service | Conteneur | URL |
|---|---|---|
| API | ml-churn-api | http://localhost:8000/docs |
| Elasticsearch | — | http://localhost:9200 |
| Logstash | — | (lit `logs/api.log`) |
| Kibana | — | http://localhost:5601 |
| Prometheus | — | http://localhost:9090 |

Grafana est lancé à part :

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

Puis http://localhost:3000 (admin / admin). Pour la source de données Prometheus, utiliser l'URL `http://host.docker.internal:9090`.

> ⚠️ Kibana et Elasticsearch mettent 1 à 2 minutes à démarrer. Attendre avant de tester.

Arrêt de la stack :

```bash
docker compose down        # conserve les données Elasticsearch
docker compose down -v     # supprime aussi le volume es-data
```

---

## 5. Utilisation de l'API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | Informations sur l'application |
| GET | `/health` | Contrôle de santé (`{"status": "UP"}`) |
| GET | `/model/info` | Modèle et version |
| POST | `/predict` | Prédiction du churn d'un client |
| GET | `/metrics` | Métriques Prometheus |
| GET | `/docs` | Documentation Swagger interactive |

Exemple de prédiction :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2
  }'
```

Réponse :

```json
{
  "prediction": "CHURN",
  "probability": 0.87,
  "model_version": "1.0.0",
  "duration_ms": 12.7
}
```

Un corps de requête invalide (champ manquant ou mauvais type) renvoie **422** grâce à la validation Pydantic.

---

## 6. Tests

### 6.1 Tests unitaires et d'API (Pytest)

```bash
source .venv/bin/activate
pytest --ignore=tests/test_selenium.py -v
```

Résultat attendu : **4 passed**. Le fichier `pytest.ini` (`pythonpath = .`) permet d'importer le module `api`.

### 6.2 Test E2E (Selenium)

L'API doit tourner sur le port 8000, et Chrome doit être installé.

```bash
pytest tests/test_selenium.py -v
```

Le test ouvre `/docs` et vérifie que le titre contient « Swagger UI ».

### 6.3 Test de charge (JMeter)

Plan : `tests/jmeter/churn_load_test.jmx` (POST `/predict`, ramp-up 10 s, 10 boucles par utilisateur). Le nombre d'utilisateurs se règle avec la propriété `threads` (100 par défaut).

Sous Windows (PowerShell), depuis le dossier `bin` de JMeter :

```powershell
.\jmeter.bat -n -t C:\jmeter-tests\churn_load_test.jmx -Jthreads=100 `
  -l C:\jmeter-tests\results_100.jtl -e -o C:\jmeter-tests\report_100
```

Le rapport HTML est généré dans `report_100\index.html`. Le dossier de sortie ne doit pas exister avant le lancement.

---

## 7. Intégration continue (GitHub Actions)

Le fichier `.github/workflows/ci.yml` s'exécute à chaque push et à chaque Pull Request :

1. checkout du code ;
2. installation de Python 3.12 ;
3. `pip install -r requirements.txt` ;
4. `pytest --ignore=tests/test_selenium.py` (Selenium exclu : pas de navigateur ni d'API lancée sur le runner).

### Workflow Git

```
feature/*  ──PR──►  develop  ──PR──►  main
```

- `main` : version stable, déployable.
- `develop` : intégration.
- `feature/model`, `feature/corrections`… : développements isolés.

Aucune fusion n'a lieu sans une CI verte.

---

## 8. Logs centralisés (ELK)

L'API écrit des logs structurés au format `clé=valeur` dans `logs/api.log` (et dans la console) :

```
2026-10-15 08:21:50,123 INFO event=prediction_success client_ip=172.18.0.1 model_version=1.0.0 prediction=CHURN probability=0.8700 duration_ms=12.70
```

Événements : `prediction_request`, `prediction_success`, `prediction_error`.

**Logstash** (`docker/logstash/logstash.conf`) lit ce fichier, puis :
- `grok` extrait l'horodatage, le niveau et le message ;
- `kv` découpe les paires `clé=valeur` en champs ;
- `date` alimente `@timestamp` ;
- `mutate` convertit `duration_ms` et `probability` en nombres.

Les données sont indexées dans **Elasticsearch** sous `ml-churn-logs-AAAA.MM.JJ`.

**Kibana** : créer une data view `ml-churn-logs-*` (champ de temps `@timestamp`) via http://localhost:5601/app/management/kibana/dataViews, puis explorer dans Discover. Exemples de filtres KQL :

```
event : "prediction_success"
prediction : "CHURN"
level : "ERROR"
duration_ms > 50
```

---

## 9. Monitoring (Prometheus + Grafana)

`prometheus-fastapi-instrumentator` expose `/metrics`. Prometheus collecte la cible `api:8000` toutes les 5 s (`monitoring/prometheus/prometheus.yml`).

Vérification : http://localhost:9090/targets → la cible `ml-churn-api` doit être **UP**.

### Dashboard Grafana

Fichier : `monitoring/grafana/ml-churn-dashboard.json`. Import : *Dashboards → New → Import → Upload JSON*.

| Panneau | Requête PromQL |
|---|---|
| Nombre de requêtes | `sum(increase(http_requests_total{handler!="/metrics"}[$__range]))` |
| Latence p95 de /predict | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{handler="/predict"}[1m])) by (le))` |
| Nombre d'erreurs | `sum(increase(http_requests_total{handler="/predict",status!="2xx"}[$__range])) or vector(0)` |
| Taux d'erreur (%) | `(sum(rate(http_requests_total{handler="/predict",status!="2xx"}[5m])) or vector(0)) / sum(rate(http_requests_total{handler="/predict"}[5m])) * 100` |
| Nombre de prédictions | `sum(increase(http_requests_total{handler="/predict",status="2xx"}[$__range]))` |

> Un résultat `NaN` dans Prometheus signifie simplement qu'il n'y a pas eu de trafic dans la fenêtre de calcul. Envoyer quelques requêtes `/predict`.

---

## 10. Déploiement sur un serveur

Procédure validée sur une distribution Ubuntu 22.04 vierge, avec un utilisateur dédié `deploy` :

```bash
sudo adduser deploy
sudo usermod -aG docker deploy
su - deploy

git clone https://github.com/christianabakar/ml-churn-project.git
cd ml-churn-project
docker compose up -d --build
docker compose ps

curl http://SERVER_IP:8000/health
curl -X POST http://SERVER_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"anciennete":5,"revenu":450000,"nb_reclamations":2,"montant_mensuel":25000,"retards_paiement":2}'
```

Mise à jour d'une version :

```bash
git pull origin main
docker compose up -d --build
```

---

## 11. Corrections apportées au TP

| # | Problème constaté | Correction | Effet mesuré |
|---|---|---|---|
| 1 | `pip install` échoue en Python 3.11 (contourpy 1.4.0 exige Python ≥ 3.12) | `python:3.12-slim` dans le Dockerfile, `3.12` dans la CI | Build Docker et CI au vert |
| 2 | `ModuleNotFoundError: api` lors de l'exécution de pytest | `pytest.ini` avec `pythonpath = .` | Tests exécutables |
| 3 | Titre Swagger réel : « Churn Prediction API - Swagger UI » | Assertion sur `"Swagger UI" in driver.title` | Test Selenium vert |
| 4 | Selenium impossible sur le runner CI | `pytest --ignore=tests/test_selenium.py` dans la CI | CI stable |
| 5 | Pas de `.dockerignore` : `.venv` (482 Mo) et `.git` copiés dans l'image | Ajout d'un `.dockerignore` | Image de 1,5 Go → 839 Mo |
| 6 | Prédiction faite sur un tableau sans noms de colonnes (warning scikit-learn), avec deux appels au modèle (`predict` + `predict_proba`) | `DataFrame` avec les colonnes `FEATURES` et un seul `predict_proba` | Warnings 3 → 1, latence 24 → 12,7 ms (−48 %) |
| 7 | Logs non exploitables par ELK et aucune gestion d'erreur | Logs structurés `clé=valeur`, écriture dans `logs/api.log`, `HTTPException` 500 tracée | Logs filtrables dans Kibana |
| 8 | Fichier `logs/api.log` créé par le conteneur en root, ce qui bloque pytest en local | `sudo chown -R $USER:$USER logs` et `logs/` ajouté au `.gitignore` | Tests verts en local |

---

## 12. Résultats des tests de charge

Machine : Windows 11, 16 Go de RAM, API dans Docker (1 worker Uvicorn), stack ELK et Prometheus active.

| Charge | Requêtes | Durée | Débit | Temps moyen | Erreurs |
|---|---|---|---|---|---|
| 100 utilisateurs | 1 000 | 1 min 34 | 10,7 req/s | 8,1 s | 0 % |
| 500 utilisateurs | 5 000 | 8 min 08 | 10,2 req/s | 45,5 s | 0 % |
| 1000 utilisateurs | 10 000 | 16 min 34 | 10,1 req/s | 93,7 s | 0 % |
| **100 utilisateurs, après corrections** | 1 000 | **1 min 10** | **14,3 req/s** (16,7 en régime établi) | **6,0 s** | 0 % |

**Analyse :**
- Le débit plafonne à environ 10 req/s quelle que soit la charge : l'API est **saturée** (un seul worker).
- Le temps de réponse croît avec le nombre d'utilisateurs, conformément à la **loi de Little** (temps d'attente ≈ utilisateurs / débit) : les requêtes font la queue.
- Le taux d'erreur de 0 % est **trompeur** : JMeter n'impose pas de timeout. En production, un délai de 90 s serait un échec.
- Les corrections augmentent le débit de **+34 %** et réduisent le temps moyen de **−26 %**.

---

## 13. Limites et pistes d'amélioration

- **Scalabilité** : lancer plusieurs workers (`uvicorn ... --workers 4`) ou plusieurs réplicas derrière un répartiteur de charge (Nginx, Kubernetes).
- **Timeouts** : configurer un délai de réponse dans JMeter et définir un SLO (par exemple p95 < 500 ms).
- **Modèle** : F1 ≈ 0,5 sur des données synthétiques. Il faudrait des données réelles, une optimisation des hyperparamètres, gérer le déséquilibre des classes et choisir le modèle sur le F1 (le Decision Tree était meilleur).
- **MLOps** : versionner les modèles (MLflow), surveiller la dérive des données et automatiser le réentraînement.
- **Sécurité** : activer la sécurité Elasticsearch (désactivée ici), ajouter une authentification sur l'API et ne pas stocker le modèle dans Git (12 Mo).
- **CD** : ajouter une étape de build et de push de l'image vers un registre, puis un déploiement automatique.
- **Déploiement** : le serveur de test partageait le moteur Docker de la machine de développement. Une vraie VM ou un serveur cloud reste à prévoir.

---

## 14. Dépannage

| Symptôme | Solution |
|---|---|
| `docker: command not found` dans WSL | Démarrer Docker Desktop et activer *Settings → Resources → WSL integration* |
| `permission denied ... docker.sock` | `sudo usermod -aG docker $USER && newgrp docker` |
| `PermissionError: logs/api.log` avec pytest | `sudo chown -R $USER:$USER logs` |
| Kibana inaccessible | Attendre 1 à 2 min. Puis `docker compose logs kibana` |
| Aucun log dans Kibana | Vérifier `docker compose logs logstash` et que `logs/api.log` contient des lignes |
| `NaN` dans Prometheus ou Grafana | Pas de trafic récent : envoyer des requêtes `/predict` |
| Grafana ne voit pas Prometheus | URL de la source de données : `http://host.docker.internal:9090` |
| `.\jmeter.bat` non reconnu | Se placer d'abord dans le dossier `bin` de JMeter (chemin entre guillemets) |

---

*Projet réalisé dans le cadre du TP fil rouge « Industrialisation des développements » — Christian NDZUNE.*
