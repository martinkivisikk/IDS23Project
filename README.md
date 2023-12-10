# IDS23Project
Repository for the Intro to Data Science project. Authors: Martin Kivisikk, Margaret Pütsepp, Ella Hiedel. Spam emails have become an issue, posing threats to individuals and organizations. This project aims to address this challenge through the development of an effective classification model. The consequences of spam emails include productivity loss, security concerns, and potential financial risks. The goal of this project is to implement and compare different machine learning models to accurately classify emails as spam or non-spam.


## Access the project online

[ids23project.onrender.com](https://ids23project.onrender.com/)

NB! Using the KNN3 model on the website will produce an out of memory error, testable on local machine only. Also, if the website has been inactive for some time, initial loading might take a bit as we are using a free hosting service.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/martinkivisikk/IDS23Project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd IDS23Project
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
    ```bash
    python nltk_setup.py
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

3. Interact with the web application.
