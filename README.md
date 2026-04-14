# 🍎 Food Nutrition AI — Pro Web Application

🔥 AI-powered food nutrition analysis system using Machine Learning + NLP
Built with Flask, scikit-learn, and modern web technologies.

---

# 🚀 Features

| Feature                      | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| 🔍 **NLP Semantic Search**   | Search foods using natural language ("high protein low fat", "vitamin C rich") |
| 🧠 **AI Caloric Predictor**  | Input custom nutrients → ML predicts calories & food category                  |
| ⚖️ **Food Comparator**       | Side-by-side nutritional comparison of any two foods                           |
| 📊 **Interactive Dashboard** | Chart.js-powered live visualizations                                           |
| 🤖 **ML Models**             | Regression, Classification & Clustering with full metrics                      |
| 🖼️ **Plot Gallery**         | 18 pre-generated matplotlib/seaborn visualizations                             |
| 📈 **EDA Explorer**          | Vitamins, minerals, macro distributions, group comparisons                     |


---

# ⚙️ Setup & Run

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run the application

```bash
python app.py
```

---

## 3️⃣ Open in browser

```
http://localhost:5000
```

---

### ⚠️ Note

On first launch:

* Models train in background (~30–60 seconds)
* UI shows loading screen until ready

---

# 🌐 Optional: Public Access (ngrok)

## 1️⃣ Start Flask app

```bash
python app.py
```

---

## 2️⃣ Start ngrok tunnel

```bash
ngrok http 5000
```

---

## 3️⃣ Open generated link

Example:

```
https://xxxx.ngrok-free.app
```

---

# 📂 Project Structure

```
Food-Nutrition-AI/
│
├── app.py                  # Flask app + API routes
├── requirements.txt
├── Combined_FOOD_METADATA.csv
│
├── data/                  # Food datasets (CSV files)
│   ├── FOOD-DATA-GROUP1.csv
│   ├── FOOD-DATA-GROUP2.csv
│   ├── FOOD-DATA-GROUP3.csv
│   ├── FOOD-DATA-GROUP5.csv
│
├── src/
│   ├── data_engine.py      # Data loading + preprocessing
│   ├── ml_models.py        # ML models (Regression, Classification, Clustering)
│   ├── eda_plots.py        # Visualization scripts
│   └── data_loader_compat.py
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/             # All 18 plots
│
└── screenshots/
```

---

# 🤖 ML Models Used

## 📊 Regression (Calorie Prediction)

* Linear Regression
* Ridge Regression
* Random Forest Regressor ⭐
* Gradient Boosting Regressor
* SVR (RBF kernel)

---

## 🏷️ Classification (Caloric Category)

* Logistic Regression
* Decision Tree
* Random Forest Classifier ⭐
* Gradient Boosting Classifier
* K-Nearest Neighbours

---

## 🔗 Clustering (Food Grouping)

* K-Means (k=5)
* PCA visualization
* Elbow method
* Silhouette score analysis

---

# 🧠 NLP Search Engine

Uses:

* TF-IDF Vectorization (bigrams)
* Cosine similarity ranking
* Nutrition-aware tagging (high protein, low calorie, etc.)
* Query expansion with nutrition synonyms

---

## 🔎 Example Queries

* `"high protein low fat"` → chicken, fish, egg whites
* `"rich in vitamin C"` → citrus, bell peppers, kiwi
* `"low carb keto friendly"` → meats, oils, leafy greens
* `"iron rich food"` → liver, spinach, lentils

---

# 🥗 Dataset

* ~2000+ food items
* 36 nutritional features

## Includes:

* Macronutrients: Calories, Fat, Protein, Carbs, Fiber, Sugars
* Vitamins: A, B1–B12, C, D, E, K
* Minerals: Calcium, Iron, Magnesium, Potassium, Zinc
* Derived: Nutrition Density, Health Score, Fat Ratios

---

