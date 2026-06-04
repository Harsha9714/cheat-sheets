# Data Science & Machine Learning Cheat Sheet

---

## Python Libraries Quick Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

---

## NumPy

# Array Creation
a = np.array([1, 2, 3])
b = np.zeros((3, 3))
c = np.ones((3, 3))
d = np.eye(3)
e = np.random.rand(3, 3)
f = np.random.randn(3, 3)
g = np.arange(0, 10, 2)
h = np.linspace(0, 1, 5)

# Array Operations
a.shape
a.reshape(3, 1)
a.T
np.dot(a, b)
np.matmul(a, b)
a * b

# Statistics
np.mean(a), np.median(a), np.std(a)
np.min(a), np.max(a), np.sum(a)
np.argmin(a), np.argmax(a)
np.percentile(a, 25)
np.corrcoef(a, b)
np.unique(a)
np.clip(a, 0, 1)

# Indexing & Slicing
a[0]
a[1:3]
a[a > 2]
a[(a > 1) & (a < 4)]

---

## Pandas

# Creating DataFrames
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
df = pd.read_csv("file.csv")
df = pd.read_excel("file.xlsx")
df = pd.read_json("file.json")
df = pd.read_sql(query, connection)

# Saving DataFrames
df.to_csv("file.csv", index=False)
df.to_excel("file.xlsx", index=False)
df.to_json("file.json")

# Exploration
df.head(10)
df.tail(10)
df.shape
df.info()
df.describe()
df.dtypes
df.columns
df.nunique()
df.value_counts()
df.sample(5)

# Selection
df["col"]
df[["col1", "col2"]]
df.iloc[0]
df.iloc[0:5, 0:3]
df.loc[0, "col"]
df.loc[df["age"] > 25]

# Cleaning
df.isnull().sum()
df.dropna()
df.dropna(subset=["col"])
df.fillna(0)
df.fillna(df.mean())
df.fillna(method="ffill")
df.drop_duplicates()
df.drop(columns=["col"])
df.rename(columns={"old": "new"})
df["col"] = df["col"].astype(int)
df["col"] = df["col"].str.strip()
df["col"] = df["col"].str.lower()

# Feature Engineering
df["new_col"] = df["a"] + df["b"]
df["log_col"] = np.log1p(df["col"])
df["binned"] = pd.cut(df["col"], bins=5)
df["rank"] = df["col"].rank()
df["lag1"] = df["col"].shift(1)
df["rolling_mean"] = df["col"].rolling(7).mean()

# Grouping & Aggregation
df.groupby("col").mean()
df.groupby("col").agg({"a": "sum", "b": "mean"})
df.groupby(["col1", "col2"]).size().reset_index(name="count")
df.pivot_table(values="val", index="row", columns="col", aggfunc="mean")

# Merging
pd.merge(df1, df2, on="id", how="inner")
pd.merge(df1, df2, on="id", how="left")
pd.concat([df1, df2], axis=0)
pd.concat([df1, df2], axis=1)

# Sorting
df.sort_values("col", ascending=False)
df.sort_values(["col1", "col2"], ascending=[True, False])

---

## Matplotlib & Seaborn

# Matplotlib Basics
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color="blue", linewidth=2, label="line")
ax.scatter(x, y, color="red", alpha=0.5, s=50)
ax.bar(x, y, color="green")
ax.hist(data, bins=30, edgecolor="black")
ax.set_title("Title", fontsize=16)
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].hist(data)
axes[1, 1].bar(x, y)

# Seaborn
sns.set_theme(style="whitegrid")
sns.histplot(df["col"], bins=30, kde=True)
sns.boxplot(x="category", y="value", data=df)
sns.violinplot(x="category", y="value", data=df)
sns.scatterplot(x="a", y="b", hue="category", data=df)
sns.lineplot(x="date", y="value", data=df)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
sns.pairplot(df, hue="target")
sns.countplot(x="category", data=df)
sns.barplot(x="category", y="value", data=df)

---

## Data Preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler = MinMaxScaler()

# Encoding
le = LabelEncoder()
df["col_encoded"] = le.fit_transform(df["col"])

ohe = OneHotEncoder(sparse=False, drop="first")
encoded = ohe.fit_transform(df[["col"]])
df = pd.get_dummies(df, columns=["col"], drop_first=True)

# Imputation
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Pipelines
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)

---

## Supervised Learning

# Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr.coef_
lr.intercept_

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

log_reg = LogisticRegression(C=1.0, max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg.predict(X_test)
log_reg.predict_proba(X_test)

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf.feature_importances_

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31
)

# Other Classifiers
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

svm = SVC(C=1.0, kernel="rbf", probability=True)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

---

## Unsupervised Learning

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Elbow Method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
plt.plot(range(1, 11), inertias, marker="o")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca.explained_variance_ratio_

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

---

## Model Evaluation

# Classification Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(y_test, y_pred_proba)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], "k--")

# Regression Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Cross Validation & Hyperparameter Tuning
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.01, 0.1, 0.3]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_

random_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, random_state=42)

# Feature Importance
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values().plot(kind="barh")

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

---

## Deep Learning (PyTorch)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNet(input_size=10, hidden_size=64, output_size=1).to(device)

criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

---

## Deep Learning (Keras/TensorFlow)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(input_size,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()

y_pred = (model.predict(X_test) > 0.5).astype(int)

---

## NLP

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus)

from transformers import pipeline, AutoTokenizer, AutoModel

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")

generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=100)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :]

---

## Time Series

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("data.csv", parse_dates=["date"], index_col="date")
df = df.asfreq("D")
df = df.fillna(method="ffill")

result = seasonal_decompose(df["value"], model="additive", period=12)
result.plot()

adf_result = adfuller(df["value"])
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

df["diff"] = df["value"].diff()

model = ARIMA(df["value"], order=(p, d, q))
result = model.fit()
forecast = result.forecast(steps=30)
result.summary()

df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek
df["lag_1"] = df["value"].shift(1)
df["lag_7"] = df["value"].shift(7)
df["rolling_mean_7"] = df["value"].rolling(7).mean()
df["rolling_std_7"] = df["value"].rolling(7).std()

---

## ML Workflow Checklist

1. Define the problem
   [ ] Classification, regression, or clustering?
   [ ] What is the target variable?
   [ ] What metric matters most?

2. Explore the data (EDA)
   [ ] df.info(), df.describe(), df.isnull().sum()
   [ ] Distribution plots for all features
   [ ] Correlation heatmap
   [ ] Target variable distribution

3. Clean the data
   [ ] Handle missing values
   [ ] Remove or fix outliers
   [ ] Fix data types
   [ ] Remove duplicates

4. Feature Engineering
   [ ] Create new features
   [ ] Encode categoricals
   [ ] Scale numerics
   [ ] Handle class imbalance (SMOTE, class_weight)

5. Model Selection
   [ ] Start simple (Linear/Logistic Regression)
   [ ] Try tree-based models (RF, XGBoost)
   [ ] Cross-validate all models
   [ ] Compare metrics fairly

6. Hyperparameter Tuning
   [ ] GridSearchCV or RandomizedSearchCV
   [ ] Tune top 2-3 models

7. Evaluate & Interpret
   [ ] Final metrics on test set
   [ ] Confusion matrix / residual plots
   [ ] Feature importances / SHAP values

8. Deploy
   [ ] Save model with joblib or pickle
   [ ] Build API with FastAPI or Flask
   [ ] Monitor performance over time

---

## Saving & Loading Models

import joblib
import pickle

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

model.save("model.h5")
model = keras.models.load_model("model.h5")

torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))

---

## Handling Imbalanced Data

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight="balanced")

from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight = dict(enumerate(weights))
model.fit(X_train, y_train, class_weight=class_weight)

---

## When to Use What

Problem                    | Algorithm
---------------------------|------------------------------------------
Binary classification      | Logistic Regression, XGBoost, Random Forest
Multi-class classification | Random Forest, XGBoost, Neural Network
Regression                 | Linear Regression, XGBoost, Random Forest
Clustering                 | K-Means, DBSCAN
Dimensionality reduction   | PCA, t-SNE, UMAP
Text classification        | TF-IDF + LR, BERT
Time series forecasting    | ARIMA, LSTM, XGBoost
Anomaly detection          | Isolation Forest, DBSCAN, Autoencoders
Recommendation             | Collaborative Filtering, Matrix Factorization
Image classification       | CNN, ResNet, EfficientNet