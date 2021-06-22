# Dependencies
import pandas as pd
import numpy as np
from flask import Flask, render_template, redirect, jsonify, request, url_for
# NLP & ML dependencies
import nltk
import string
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
# import joblib
# import pickle
# from sklearn.metrics import classification_report


app = Flask(__name__)


# Routes to render templates
# ---------------------------------
# ---------------------------------


# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods =['POST'])
def predict():

    """
    create model and pipeline
    """
    movies_df = pd.read_csv("https://data-bootcamp-ztc.s3.amazonaws.com/movies_complete_cleaned.csv")
    movies_df = movies_df[movies_df["rating"] != "NC-17"]
    df = movies_df[["name", "plot", "rating"]]
    df.set_index("name", inplace=True)
    df = df.dropna(axis='index', subset=['plot'])

    # Penerate an additional feature using plot's length
    df["plot_len"] = [len(x) for x in df["plot"]]

    # Remove punctuations
    def remove_punct(text):
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)
    df["plot"] = [remove_punct(x) for x in df["plot"]]

    # Remove stop words
    nltk.download('stopwords')
    stop = set(stopwords.words("english"))
    def remove_stopwords(text):
        text = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(text)
    df["plot"] = [remove_stopwords(x) for x in df["plot"]]

    # Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        text = [wordnet_lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)
    df["plot"] = [lemmatize(x) for x in df["plot"]]

    # Encode ratings
    label_encoder = preprocessing.LabelEncoder()
    df["encoded_rating"] = label_encoder.fit_transform(df["rating"])
    
    # Genre
    genres = pd.read_csv("https://data-bootcamp-ztc.s3.amazonaws.com/parsed_genres_table.csv")  # one hot encoding genres
    genres.set_index("name", inplace = True)
    genres = genres.drop(columns = ["genre_kaggle", "genres_omdb"])
    clean_df = pd.merge(df, genres, how = "inner", on = "name")

    #split dataset
    X = clean_df.drop(columns = ["rating", "encoded_rating"])
    y = clean_df["encoded_rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # create tfidf vector
    cv = HashingVectorizer().fit(X_train["plot"]) #hasher
    X_train_counts = cv.transform(X_train["plot"])
    X_test_counts = cv.transform(X_test["plot"])
    tf_idf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts) #transfromer
    X_train_tfidf = tf_idf_transformer.transform(X_train_counts)
    X_test_tfidf = tf_idf_transformer.transform(X_test_counts)

    # Scale plot length
    plot_len_scaler = MinMaxScaler().fit(X_train["plot_len"].values.reshape(-1,1))
    X_train["plot_len"] = plot_len_scaler.transform(X_train["plot_len"].values.reshape(-1,1))
    X_test["plot_len"] = plot_len_scaler.transform(X_test["plot_len"].values.reshape(-1,1))

    # Create extra features matrix
    X_train_extra_features_mat = csr_matrix(X_train.drop(columns=["plot"])) 
    X_test_extra_features_mat = csr_matrix(X_test.drop(columns=["plot"]))  

    # Combine the matrices
    X_train_full = hstack([X_train_tfidf, X_train_extra_features_mat])
    X_test_full = hstack([X_test_tfidf, X_test_extra_features_mat])

    # fit the model 
    clf = LinearSVC(class_weight='balanced',verbose=2, random_state=42,max_iter=100000)
    clf.fit(X_train_full, y_train)

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
    print(request.get_json())  # parse as JSON

    # Get POST'ed user data as a JSON object
    data = request.get_json()


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

    # Compare input matrix shape with X_train shape
    print("----------------------")
    print("X_train matrix shape:")
    print(X_train_full.shape)
    print("Input matrix shape:")
    print(input_mat.shape)

    output = int(clf.predict(input_mat))

    print("----------------------")
    print("Prediction (0:G, 1:PG, 2:PG-13, 3:R):")
    print(output)

    # # Calculate classification report
    # predictions = clf.predict(X_test_full)
    # target_names = ["G", "PG", "PG-13", "R"]
    # print(classification_report(y_test, predictions,
    #                             target_names=target_names))

    print("-----------------------------")
    print("PREDICTION SUCCESSFUL")
    print("-----------------------------")

    return jsonify(output)




if __name__ == '__main__':
    app.run(debug=True)
