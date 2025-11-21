

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def initialize_nltk():
    """Download necessary NLTK datasets on first run"""
    required = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in required:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


initialize_nltk()


class TextPreprocessor:

    def __init__(self, remove_stopwords=True, use_lemmatization=True):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # preserve sentiment-critical words
        sentiment_markers = {'not', 'no', 'nor', 'neither', 'never', 'none',
                             'very', 'really', 'quite', 'much', 'too', 'but'}
        self.stop_words -= sentiment_markers

    def clean_text(self, text):

        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        text = ' '.join(text.split())

        return text

    def tokenize(self, text):

        return word_tokenize(text)

    def preprocess(self, text):

        text = self.clean_text(text)

        if not text:
            return ""

        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    def preprocess_batch(self, texts, show_progress=False):

        processed = []
        total = len(texts)

        for idx, text in enumerate(texts):
            processed.append(self.preprocess(text))

            if show_progress and (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{total} texts")

        return processed


class DatasetManager:
    """
     CSV loading and automatic labeling using TextBlob.
    """

    def __init__(self):
        self.data = None
        self.preprocessor = TextPreprocessor()

    def load_from_csv(self, filepath, text_column, label_column):
        """Load dataset from CSV file"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Loaded {len(self.data)} samples from {filepath}")

            self.data = self.data[[text_column, label_column]].copy()
            self.data.columns = ['text', 'sentiment']

            print("\nLabel distribution:")
            print(self.data['sentiment'].value_counts())

            return self.data

        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def create_from_arrays(self, texts, labels):

        self.data = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })

        print(f"Dataset created with {len(self.data)} samples")
        return self.data

    def auto_label_textblob(self, texts):

        print(f"Auto-labeling {len(texts)} texts using TextBlob...")

        labels = []
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                labels.append('positive')
            elif polarity < -0.1:
                labels.append('negative')
            else:
                labels.append('neutral')

        distribution = Counter(labels)
        print("Label distribution:")
        for label, count in distribution.items():
            print(f"  {label}: {count}")

        return labels

    def clean_dataset(self):

        if self.data is None:
            print("No dataset loaded")
            return None

        print("\nCleaning dataset...")

        # remove duplicates and empty entries
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['text'])
        self.data = self.data[self.data['text'].notna()]
        self.data = self.data[self.data['text'].str.strip() != '']

        print(f"Removed {initial_size - len(self.data)} invalid entries")

        print("Preprocessing text...")
        self.data['processed_text'] = self.preprocessor.preprocess_batch(
            self.data['text'].tolist(),
            show_progress=True
        )

        self.data = self.data[self.data['processed_text'].str.strip() != '']

        print(f"Final dataset size: {len(self.data)} samples")
        return self.data

    def save_dataset(self, filepath='sentiment_dataset.csv'):

        if self.data is not None:
            self.data.to_csv(filepath, index=False)
            print(f"Dataset saved to {filepath}")

    def load_dataset(self, filepath='sentiment_dataset.csv'):

        try:
            self.data = pd.read_csv(filepath)
            print(f"Dataset loaded from {filepath}")
            return self.data
        except:
            print(f"Could not load dataset from {filepath}")
            return None


class SentimentClassifier:
    """
    Sentiment classification model using supervised learning.
    Supports multiple algorithms: Logistic Regression, Naive Bayes, SVM.
    """

    def __init__(self, algorithm='logistic', vectorizer='tfidf'):
        self.algorithm = algorithm
        self.vectorizer_type = vectorizer

        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000, ngram_range=(1, 2))

        classifiers = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': LinearSVC(random_state=42, max_iter=2000)
        }

        self.model = classifiers.get(algorithm, classifiers['logistic'])
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the sentiment classifier.
        Optionally validate on separate validation set.
        """
        print(f"\nTraining {self.algorithm} classifier...")
        print(f"Training samples: {len(X_train)}")

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"Feature matrix shape: {X_train_vec.shape}")

        self.model.fit(X_train_vec, y_train_encoded)

        train_pred = self.model.predict(X_train_vec)
        train_acc = accuracy_score(y_train_encoded, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_vec = self.vectorizer.transform(X_val)
            val_pred = self.model.predict(X_val_vec)
            val_acc = accuracy_score(y_val_encoded, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")

        self.is_trained = True

    def predict(self, texts):

        if not self.is_trained:
            raise ValueError("Model has not been trained")

        if isinstance(texts, str):
            texts = [texts]

        X_vec = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_vec)
        labels = self.label_encoder.inverse_transform(predictions)

        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_vec)

        return labels, probabilities

    def predict_single(self, text):

        labels, probs = self.predict(text)

        result = {
            'text': text,
            'sentiment': labels[0]
        }

        if probs is not None:
            result['confidence'] = float(np.max(probs[0]))
            result['probabilities'] = {
                self.label_encoder.classes_[i]: float(probs[0][i])
                for i in range(len(self.label_encoder.classes_))
            }

        return result

    def evaluate(self, X_test, y_test, plot_confusion=True):

        if not self.is_trained:
            raise ValueError("Model has not been trained")

        print("\nEvaluating model...")

        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)

        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test_encoded,
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))

        if plot_confusion:
            cm = confusion_matrix(y_test_encoded, y_pred)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()

        return accuracy

    def save_model(self, filepath='sentiment_model.pkl'):

        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'algorithm': self.algorithm,
            'vectorizer_type': self.vectorizer_type
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='sentiment_model.pkl'):

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.algorithm = model_data['algorithm']
            self.vectorizer_type = model_data['vectorizer_type']
            self.is_trained = True

            print(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def run_demo():

    print("="*60)
    print("Sentiment Analysis Demo")
    print("="*60)

    # sample dataset
    sample_texts = [
        "This product is amazing! Best purchase ever.",
        "Terrible quality, very disappointed with this.",
        "It's okay, nothing special really.",
        "Absolutely love it! Highly recommend to everyone.",
        "Worst experience I've ever had.",
        "Pretty decent for the price point.",
        "Outstanding quality and fast delivery!",
        "Not satisfied at all with this product.",
        "Works as expected, no complaints.",
        "Fantastic! Exceeded all my expectations."
    ]

    sample_labels = [
        'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral', 'positive'
    ]

    manager = DatasetManager()
    dataset = manager.create_from_arrays(sample_texts, sample_labels)
    dataset = manager.clean_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset['processed_text'],
        dataset['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=dataset['sentiment']
    )

    classifier = SentimentClassifier(algorithm='logistic', vectorizer='tfidf')
    classifier.train(X_train, y_train)

    classifier.evaluate(X_test, y_test, plot_confusion=False)

    print("\n" + "="*60)
    print("Testing Predictions")
    print("="*60)

    test_examples = [
        "I absolutely love this product!",
        "This is terrible, complete waste of money.",
        "It's alright, nothing extraordinary."
    ]

    for text in test_examples:
        result = classifier.predict_single(text)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {result['sentiment']}")
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2%}")

    print("\n" + "="*60)

    return classifier, manager


def main():

    print("="*60)
    print("Sentiment Analysis System v1.0")
    print("="*60)

    classifier = SentimentClassifier()
    manager = DatasetManager()

    while True:
        print("\nOptions:")
        print("1. Load dataset from CSV")
        print("2. Create dataset manually")
        print("3. Auto-label data with TextBlob")
        print("4. Clean and preprocess dataset")
        print("5. Train classifier")
        print("6. Evaluate classifier")
        print("7. Predict sentiment (single text)")
        print("8. Batch prediction")
        print("9. Save model")
        print("10. Load model")
        print("11. Run demo")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == '1':
            filepath = input("CSV file path: ").strip()
            text_col = input("Text column name: ").strip()
            label_col = input("Label column name: ").strip()
            manager.load_from_csv(filepath, text_col, label_col)

        elif choice == '2':
            num_samples = int(input("Number of samples: ").strip())
            texts, labels = [], []
            for i in range(num_samples):
                text = input(f"Text {i+1}: ").strip()
                label = input(f"Label: ").strip()
                texts.append(text)
                labels.append(label)
            manager.create_from_arrays(texts, labels)

        elif choice == '3':
            if manager.data is None:
                print("No dataset loaded")
            else:
                labels = manager.auto_label_textblob(
                    manager.data['text'].tolist())
                manager.data['sentiment'] = labels

        elif choice == '4':
            manager.clean_dataset()

        elif choice == '5':
            if manager.data is None or 'processed_text' not in manager.data.columns:
                print("Dataset not ready. Load and clean data first.")
            else:
                test_size = float(
                    input("Test set size (0-1): ").strip() or "0.2")

                X_train, X_test, y_train, y_test = train_test_split(
                    manager.data['processed_text'],
                    manager.data['sentiment'],
                    test_size=test_size,
                    random_state=42
                )

                classifier.train(X_train, y_train, X_test, y_test)

        elif choice == '6':
            if not classifier.is_trained:
                print("Classifier not trained")
            elif manager.data is None:
                print("No dataset available")
            else:
                _, X_test, _, y_test = train_test_split(
                    manager.data['processed_text'],
                    manager.data['sentiment'],
                    test_size=0.2,
                    random_state=42
                )
                classifier.evaluate(X_test, y_test)

        elif choice == '7':
            text = input("Enter text: ").strip()
            if classifier.is_trained:
                preprocessor = TextPreprocessor()
                processed = preprocessor.preprocess(text)
                result = classifier.predict_single(processed)

                print(f"\nSentiment: {result['sentiment']}")
                if 'confidence' in result:
                    print(f"Confidence: {result['confidence']:.2%}")
                    print("\nProbabilities:")
                    for label, prob in result['probabilities'].items():
                        print(f"  {label}: {prob:.2%}")
            else:
                print("Classifier not trained")

        elif choice == '8':
            filepath = input("Text file path (one text per line): ").strip()
            try:
                with open(filepath, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]

                preprocessor = TextPreprocessor()
                processed = [preprocessor.preprocess(t) for t in texts]
                labels, _ = classifier.predict(processed)

                for text, label in zip(texts, labels):
                    print(f"{text[:60]}... -> {label}")
            except:
                print("Error reading file")

        elif choice == '9':
            filepath = input("Save to (default sentiment_model.pkl): ").strip()
            classifier.save_model(filepath or 'sentiment_model.pkl')

        elif choice == '10':
            filepath = input(
                "Load from (default sentiment_model.pkl): ").strip()
            classifier.load_model(filepath or 'sentiment_model.pkl')

        elif choice == '11':
            run_demo()

        elif choice == '0':
            print("\nExiting...")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
