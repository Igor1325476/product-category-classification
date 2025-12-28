# train_model_pipeline.py
import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Folder za modele ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# --- Brisanje svih postojećih .pkl fajlova ---
for f in glob.glob(os.path.join(model_dir, "*.pkl")):
    os.remove(f)
print("Sve postojeće .pkl datoteke u 'model' folderu su obrisane.")

# --- Učitavanje podataka ---
url = "https://raw.githubusercontent.com/Igor1325476/product-category-classification/main/data/IMLP4_TASK_03-products.csv"
df = pd.read_csv(url)
df.columns = [col.strip() for col in df.columns]
df = df.dropna(subset=['Product Title', 'Category Label'])

# --- Feature engineering ---
df['title_length'] = df['Product Title'].astype(str).str.len()
df['title_word_count'] = df['Product Title'].astype(str).str.split().apply(len)
df['has_number'] = df['Product Title'].astype(str).apply(lambda x: int(any(c.isdigit() for c in x)))

# --- Features i label ---
X = df[['Product Title', 'title_length', 'title_word_count', 'has_number']]
y = df['Category Label']

# --- Preprocessing i pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('title', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), 'Product Title'),
        ('length', MinMaxScaler(), ['title_length', 'title_word_count', 'has_number'])
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])

# --- Treniranje modela ---
pipeline.fit(X, y)

# --- Čuvanje novog modela ---
model_path = os.path.join(model_dir, "product_category_model_pipeline.pkl")
joblib.dump(pipeline, model_path)

print(f"Model treniran i sačuvan kao '{model_path}'")
