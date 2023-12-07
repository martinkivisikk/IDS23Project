import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# May need to download some packages from NLTK, such as stopwords, wordnet.
# Uncomment the following lines to download the packages.
# import nltk
# nltk.download("stopwords")
# nltk.download("wordnet")

"""
    Text Cleaning: Remove any unnecessary characters, symbols, or HTML tags.

    Tokenization: Split the text into individual words or tokens.

    Lowercasing: Convert all text to lowercase to ensure consistency.

    Stopword Removal: Remove common words that don't carry much information (e.g., "and", "the", "is").

    Stemming or Lemmatization: Reduce words to their base or root form.

    Vectorization: Convert the text data into a numerical format. This can be done using techniques like Bag-of-Words or TF-IDF.
"""
# Load your dataset
data = pd.read_csv('../data/combined_data.csv')

# Step 1: Text Cleaning
data['cleaned_text'] = data['text'].apply(lambda x: ' '.join(word for word in x.split()))

# Step 2: Tokenization
data['tokenized_text'] = data['cleaned_text'].apply(lambda x: x.split())

# Step 3: Lowercasing
data['tokenized_text'] = data['tokenized_text'].apply(lambda x: [word.lower() for word in x])

# Step 4: Stopword Removal (you may need to download the stopword list)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
data['tokenized_text'] = data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Step 5: Stemming or Lemmatization (you may need to download a lemmatizer)
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
data['tokenized_text'] = data['tokenized_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Step 6: Vectorization (choose either CountVectorizer or TfidfVectorizer)
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features based on your dataset size
X = vectorizer.fit_transform(data['tokenized_text'].apply(lambda x: ' '.join(x)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# Train a model (using Naive Bayes as an example)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer to pkl files
joblib.dump(model, 'NaiveBayes.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Naive Bayes: {accuracy:.2f}')

# You can also print other metrics like classification report
print(classification_report(y_test, y_pred))

"""
LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(X_train, y_train)
joblib.dump(LogisticRegressionModel, 'LogisticRegression.pkl')

y_pred = LogisticRegressionModel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of LogisticRegressionModel: {accuracy:.2f}')


model_KNN3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model_KNN3.fit(X_train, y_train)
y_pred = model_KNN3.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN3: {accuracy:.2f}')

joblib.dump(model_KNN3, 'KNN3.pkl')
"""
