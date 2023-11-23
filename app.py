import joblib
from flask import Flask, render_template, request
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

# Define the available models and their paths
model_paths = {
    'NaiveBayes': 'models/NaiveBayes.pkl',
    # Add more models as needed
}

# Load your trained model and vectorizer
models = {}
for model_name, model_path in model_paths.items():
    try:
        models[model_name] = joblib.load(model_path)
    except Exception as e:
        app.logger.error(f"Error loading model '{model_name}': {str(e)}")

vectorizer_path = 'models/vectorizer.pkl'
try:
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    app.logger.error(f"Error loading vectorizer: {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            email_text = request.form['email_text']
            selected_model = request.form['model']

            # Validate and sanitize user input
            if not email_text:
                raise ValueError("Email text is required.")

            # Perform any necessary preprocessing on the input text
            cleaned_text = ' '.join(word for word in email_text.split())
            tokenized_text = cleaned_text.split()
            lowercased_text = [word.lower() for word in tokenized_text]
            stop_words = set(stopwords.words('english'))
            filtered_text = [word for word in lowercased_text if word not in stop_words]
            lemmatizer = WordNetLemmatizer()
            preprocessed_text = ' '.join([lemmatizer.lemmatize(word) for word in filtered_text])

            # Vectorize the preprocessed text
            input_vector = vectorizer.transform([preprocessed_text])

            # Make prediction and get probability
            if selected_model not in models:
                raise ValueError(f"Selected model '{selected_model}' is not available.")

            prediction = models[selected_model].predict(input_vector)
            probability = models[selected_model].predict_proba(input_vector)[:, 1]  # Assuming binary classification

            # Convert prediction to "Spam" or "Not Spam"
            prediction_label = "Spam" if prediction[0] == 1 else "Not spam"
            probability = 1 - probability[0] if prediction[0] == 0 else probability[0]

            # Display the prediction and probability
            return render_template('index.html', prediction=prediction_label, probability=probability,
                                   email_text=email_text, selected_model=selected_model, model_names=model_paths.keys())

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")

    return render_template('index.html', prediction=None, probability=None, email_text="", selected_model=None,
                           model_names=model_paths.keys())


if __name__ == '__main__':
    app.run(debug=True)
