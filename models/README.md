# Modèle de prédiction du churn

- Algorithme : Random Forest (200 arbres, random_state=42)
- Données : data/raw/churn.csv (2000 clients, taux de churn 43 %)
- Split : 80 % train / 20 % test, stratifié
- Scores sur le jeu de test :
  - Régression logistique : F1 = 0,510
  - Arbre de décision : F1 = 0,545
  - Random Forest : F1 = 0,487
- Fichier : models/model.pkl (version 1.0.0)

