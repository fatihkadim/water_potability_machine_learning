import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix
from sklearn import tree

# Veri setini yükleme
df = pd.read_csv("water_potability.csv")
print(df.head())
print(df.isnull().sum())

# Verinin temel istatistiklerini inceleme
some_infs = pd.DataFrame(df.describe().T)

# Potability sütununun dağılımını görselleştirme
d = pd.DataFrame(df["Potability"].value_counts()).reset_index()
d.columns = ['Potability', 'count']

fig = px.pie(d, values="count", names=["Not Potable", "Potable"], hole=0.35, opacity=0.8,
             labels={"Label": "Potability", "Potability": "Number of samples"})
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()
fig.write_html("potability_pie_chart.html")

# Korelasyon analizi
sns.clustermap(df.corr(), cmap="coolwarm", dendrogram_ratio={0.1, 0.2}, annot=True, linewidths=0.8, figsize=(10, 10))
plt.show()

# Özelliklerin dağılımını görselleştirme
non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")

plt.figure()
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3, 3, ax + 1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col], label="Non Potable")
    sns.kdeplot(x=potable[col], label="Potable")
    plt.legend()
plt.tight_layout()
plt.show()

# Eksik verileri görselleştirme
msno.matrix(df)
plt.show()

# Eksik verilerin doldurulması
ph_mean_potability_0 = df[df["Potability"] == 0]["ph"].mean()
ph_mean_potability_1 = df[df["Potability"] == 1]["ph"].mean()

df["ph"] = df.apply(lambda row: ph_mean_potability_0 if pd.isna(row["ph"]) and row["Potability"] == 0 
                      else (ph_mean_potability_1 if pd.isna(row["ph"]) and row["Potability"] == 1 
                            else row["ph"]), axis=1)

df["Sulfate"].fillna(value=df["Sulfate"].mean(), inplace=True)
df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(), inplace=True)

# Veriyi eğitim ve test olarak ayırma
X = df.drop("Potability", axis=1).values
y = df["Potability"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Min-Max normalizasyonu
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min) / (x_train_max - x_train_min)

x_test_max = np.max(X_test)
x_test_min = np.min(X_test)
X_test = (X_test - x_test_min) / (x_test_max - x_test_min)

# Model eğitimi ve değerlendirme
models = [("DTC", DecisionTreeClassifier(max_depth=3)), ("RFC", RandomForestClassifier())]
finalResults = []  # Model skorları
cmlist = []  # Confusion matrix listesi

for name, model in models:
    model.fit(X_train, y_train)
    model_result = model.predict(X_test)
    score = precision_score(y_test, model_result)
    finalResults.append((name, score))
    cm = confusion_matrix(y_test, model_result)
    cmlist.append((name, cm))

# Confusion matrix görselleştirme
for name, cm in cmlist:
    plt.figure()
    sns.heatmap(cm, annot=True, linewidth=0.8, fmt=".0f")
    plt.title(name)
    plt.show()

# Karar ağacı modelini görselleştirme
dt_clf = models[0][1]
plt.figure(figsize=(25, 20))
tree.plot_tree(dt_clf, feature_names=df.columns.tolist()[:-1], class_names=["0", "1"],
               filled=True, precision=5)
plt.show()

# Model optimizasyonu için hiperparametre arama
model_params = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_features": ["sqrt", "log2"],
            "max_depth": list(range(1, 21, 3))
        }
    }
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
scores = []

for model_name, params in model_params.items():
    rs = RandomizedSearchCV(params["model"], params["params"], cv=cv, n_iter=10)
    rs.fit(X, y)
    scores.append([model_name, dict(rs.best_params_), rs.best_score_])

print(scores)
