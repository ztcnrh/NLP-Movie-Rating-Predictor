# Dependencies
import pandas as pd
import numpy as np
from flask import Flask, render_template, redirect, jsonify, request, url_for
# NLP & ML dependencies
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import joblib
import pickle


app = Flask(__name__)


# Routes to render templates
# ---------------------------------
# ---------------------------------

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route("/predict", methods =['POST'])
def predict():

    """
    Create NLP words preprocessing pipeline for plot data only
    """

    # Function to remove punctuations
    def remove_punct(text):
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)

    # Function to remove stop words
    """
    Uncomment the below line if running locally!
    """
    # nltk.download('stopwords') # Download `stopwords`
    
    stop = set(stopwords.words("english"))
    def remove_stopwords(text):
        text = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(text)

    # Functiona to group together inflected forms of words in plot (Lemmatization)
    """
    Uncomment the below line if running locally!
    """
    # nltk.download('wordnet') # Download `wordnet`

    wordnet_lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        text = [wordnet_lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)

    """
    End of pipeline
    """ 


    """
    ------------------------------------------------------------
    Get data from JavaScript and vectorize --> transform for tf-idf --> scale --> predict
    ------------------------------------------------------------
    """ 

    # POST request
    print("----------------------")
    print("User selected inputs:")
    print(request.get_json()) # parse as JSON and visualize user selected inputs

    # Get POST'ed user data as a JSON object
    data = request.get_json()

    # Load pickled vectorizer and tfidf transformer to vectorize our input plot data
    cv = pickle.load(open("flask_app/deployed_SVM_ratings_vectorizer.pickle", "rb"))
    tf_idf_transformer = pickle.load(open("flask_app/deployed_SVM_ratings_transformer.pickle", "rb"))

    # Load saved MinMax scaler for fitted for the plot length feature
    scaler = "flask_app/deployed_SVM_ratings_scaler.sav"
    plot_len_scaler = joblib.load(scaler)

    # Preprocess 
    if data["input"]["plot"]:
        data = data["input"]
        plot = data["plot"]
        plot_len = len(plot)
        plot = remove_punct(plot)
        plot = remove_stopwords(plot)
        plot = lemmatize(plot)
        hashed_vec = cv.transform([plot])
        tfidf_vec = tf_idf_transformer.transform(hashed_vec)
        plot_len_scaled = plot_len_scaler.transform(np.array(plot_len).reshape(-1,1))

    # Create df for all non-plot data to transform them into a matrix
    extra_features_df = pd.DataFrame({
        "plot_len" : plot_len_scaled[0][0],
        "action": data["action"],
        "adventure": data["adventure"],
        "fantasy"	: data["fantasy"],
        "sci-fi"	: data["sci-fi"],            
        "crime"	: data["crime"],
        "drama"	: data["drama"],
        "history" : data["history"],
        "comedy"	: data["comedy"],
        "biography"	: data["biography"],
        "romance"	: data["romance"],
        "horror"	: data["horror"],
        "thriller"	: data["thriller"],
        "war"	: data["war"],
        "animation"	: data["animation"],
        "family"	: data["family"],
        "sport" : data["family"],
        "music" : data["music"],
        "mystery"	: data["mystery"],
        "short" : data["short"],
        "western" : data["western"],
        "musical"	: data["musical"],
        "documentary" : data["documentary"],
        "film-noir" : data["film-noir"],
        "adult" : data["adult"],
        }, index = [0])

    # Make into matrix
    extra_features_mat = csr_matrix(extra_features_df)

    # Combine with vectorized plot matrix
    input_mat = hstack([tfidf_vec, extra_features_mat])

    # Check input matrix shape
    print("----------------------")
    print("Input matrix shape:")
    print(input_mat.shape)

    # Load trained model for predictions
    classifer = "flask_app/deployed_SVM_ratings_classifier.sav"
    clf = joblib.load(classifer)
    
    # Predict outcome using user selected inputs
    output = int(clf.predict(input_mat))

    print("----------------------")
    print("Prediction (0:G, 1:PG, 2:PG-13, 3:R):")
    print(output)


    print("-----------------------------")
    print("PREDICTION SUCCESSFUL")
    print("-----------------------------")

    # Return predicted outcome to JavaScript fetch
    return jsonify(output)




if __name__ == '__main__':
    app.run(debug=True)
