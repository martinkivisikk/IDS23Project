from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model
#spam_classifier_model = joblib.load('path/to/your/model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form['email_text']

        # Perform any necessary preprocessing on the input text (similar to what you did during training)

        # Make prediction
        #prediction = spam_classifier_model.predict([email_text])
        prediction = [0,1]
        # Display the prediction
        return render_template('index.html', prediction=prediction[0], email_text=email_text)

    return render_template('index.html', prediction=None, email_text=None)

if __name__ == '__main__':
    app.run(debug=True)
