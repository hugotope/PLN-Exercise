import os
import joblib
import collections
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocess import preprocess  # tu función de preprocess


def load_data(path=None):
    """Carga dataset CSV con columnas 'message' y 'label'"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.csv")
    df = pd.read_csv(path)
    return df['message'].tolist(), df['label'].tolist()


def train_and_evaluate():
    messages, labels = load_data()
    print("Dataset size:", len(messages))
    print("Distribución de clases:", collections.Counter(labels))

    # Preprocesar mensajes
    X_text = [preprocess(m) for m in messages]

    # Vectorización TF-IDF con mejoras
    vectorizer = TfidfVectorizer(
        min_df=2,       # ignorar palabras raras
        max_df=0.9,     # ignorar palabras muy frecuentes
        ngram_range=(1, 2)  # unigramas + bigramas
    )
    X = vectorizer.fit_transform(X_text)
    y = labels

    # División entrenamiento/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Clasificadores a evaluar
    classifiers = {
        'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(max_iter=3000, class_weight='balanced'),
        'LinearSVC': LinearSVC(class_weight='balanced')
    }

    best_clf = None
    best_score = -1
    results = {}

    # Entrenamiento y evaluación
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            score = report.get('macro avg', {}).get('f1-score', 0)
            results[name] = report

            print(f"\n--- {name} ---")
            print(classification_report(y_test, y_pred))

            if score > best_score:
                best_score = score
                best_clf = clf
                best_pred = y_pred
        except Exception as e:
            print(f"Error training {name}:", e)

    # Guardar modelo y vectorizador
    out_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(out_dir, 'sallexa_model.pkl')
    vec_path = os.path.join(out_dir, 'vectorizer.pkl')
    joblib.dump(best_clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print("\nGuardado modelo en", model_path)
    print("Guardado vectorizador en", vec_path)

    # Guardar reporte resumido
    with open(os.path.join(out_dir, 'train_report.txt'), 'w', encoding='utf8') as f:
        f.write(f"best_model: {type(best_clf).__name__}\n")
        f.write(f"best_f1_macro: {best_score}\n")
        f.write(f"classes_distribution: {collections.Counter(labels)}\n")

    # Guardar matriz de confusión
    cm = confusion_matrix(y_test, best_pred, labels=list(sorted(set(y))))
    df_cm = pd.DataFrame(cm, index=sorted(set(y)), columns=sorted(set(y)))
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    df_cm.to_csv(cm_path)
    print("Guardada matriz de confusión en", cm_path)

    # Guardar imagen de matriz de confusión
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    print("Guardada imagen de matriz de confusión")

if __name__ == '__main__':
    train_and_evaluate()
