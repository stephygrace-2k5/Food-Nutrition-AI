# 🍎 Food Nutrition AI — Pro Web Application

A full-stack AI-powered food nutrition analysis platform built with Flask, scikit-learn, and modern web technologies.

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **NLP Semantic Search** | Search foods using natural language ("high protein low fat", "vitamin C rich") |
| 🧠 **AI Caloric Predictor** | Input custom nutrients → ML predicts calories & food category |
| ⚖️ **Food Comparator** | Side-by-side nutritional comparison of any two foods |
| 📊 **Interactive Dashboard** | Chart.js-powered live visualizations |
| 🤖 **ML Models** | Regression, Classification & Clustering with full metrics |
| 🖼️ **Plot Gallery** | 18 pre-generated matplotlib/seaborn visualizations |
| 📈 **EDA Explorer** | Vitamins, minerals, macro distributions, group comparisons |

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

> **Note:** On first launch, models train in the background (~30–60s depending on hardware). The UI shows a loading screen and becomes interactive once ready.

## Project Structure

```
food_nutrition_ai/
├── app.py                  # Flask app + API routes
├── requirements.txt
├── data/                   # 5 food group CSVs
│   ├── FOOD-DATA-GROUP1.csv
│   └── ...
├── src/
│   ├── data_engine.py      # Data loading + NLP index (TF-IDF)
│   ├── ml_models.py        # Regression, Classification, Clustering
│   ├── eda_plots.py        # 9 EDA matplotlib plots
│   └── data_loader_compat.py
├── templates/
│   └── index.html          # Full single-page UI
└── outputs/
    ├── plots/              # 18 PNG visualizations (auto-generated)
    └── reports/            # CSV model results (auto-generated)
```

## ML Models Used

**Regression** (predict Caloric Value):
- Linear Regression, Ridge Regression
- Random Forest Regressor ⭐
- Gradient Boosting Regressor
- SVR (RBF kernel)

**Classification** (predict Caloric Category: Low/Medium/High/Very High):
- Logistic Regression
- Decision Tree
- Random Forest Classifier ⭐
- Gradient Boosting Classifier
- K-Nearest Neighbours

**Clustering** (unsupervised food grouping):
- K-Means (k=5) with PCA visualization
- Elbow method + Silhouette score analysis

## NLP Search

The NLP search engine uses:
- **TF-IDF vectorization** with bigrams
- **Cosine similarity** ranking
- **Nutrition-aware document enrichment** — each food gets tagged with nutritional properties (e.g., "high protein", "low calorie", "rich in calcium")
- **Query expansion** with nutritional synonyms

Example queries:
- `"high protein low fat"` → chicken, fish, egg whites
- `"rich in vitamin C"` → citrus, bell peppers, kiwi
- `"low carb keto friendly"` → meats, oils, leafy greens
- `"iron rich food"` → liver, spinach, lentils

## Dataset

5 food groups combined (~2,000+ unique foods), each with 36 nutritional features:
- Macronutrients: Calories, Fat, Protein, Carbohydrates, Fiber, Sugars
- Vitamins: A, B1–B12, C, D, E, K
- Minerals: Calcium, Iron, Magnesium, Potassium, Zinc, and more
- Derived features: Nutrition Density, Health Score, Sat/Unsat Fat Ratio
