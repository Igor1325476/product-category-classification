# test_model.py
import joblib
import pandas as pd

# Učitavanje sačuvanog modela
model = joblib.load("model/product_category_model_pipeline.pkl")
print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Feature engineering za unos korisnika
    title_length = len(title)
    title_word_count = len(title.split())
    has_number = int(any(c.isdigit() for c in title))

    # Kreiranje DataFrame-a za pipeline
    user_input = pd.DataFrame([{
        "Product Title": title,
        "title_length": title_length,
        "title_word_count": title_word_count,
        "has_number": has_number
    }])

    # Predikcija kategorije
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-"*40)
