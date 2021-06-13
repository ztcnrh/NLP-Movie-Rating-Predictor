# Dependencies
import pandas as pd
import numpy as np
from os import environ
from flask import Flask, render_template, redirect, jsonify, request, url_for


app = Flask(__name__)

# import joblib
# joblib.dump(clf, 'ratings_SVM_classifier.sav')
# model = open('/Users/daniellepintacasi/finalproject/flask_app/ratings_SVM_classifier.sav','rb')
# clf = joblib.load(model)


# import joblib

# model = joblib.dump(model, "ratings_SVM_classifier.sav")


# Routes to render templates
# ---------------------------------
# ---------------------------------

# import joblib

# loaded_model = joblib.load('rating_linear_svc_with_genre.sav')

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods =['GET','POST'])
def predict():

    """
    create model and pipeline
    """
    movies_df = pd.read_csv("https://data-bootcamp-ztc.s3.amazonaws.com/movies_complete_cleaned.csv")
    movies_df = movies_df[movies_df["rating"]  != "NC-17"]
    df = movies_df[["name", "plot", "rating"]]
    df.set_index("name",inplace = True)
    df = df.dropna(axis='index', subset=['plot'])

     #generate plot len
    df["plot_len"] = [len(x) for x in df["plot"]]

    #remove punctuation 
    import string
    def remove_punct(text):
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)
    df["plot"] = [remove_punct(x) for x in df["plot"]]

    #remove stop words
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    def remove_stopwords(text):
        text = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(text)
    df["plot"] = [remove_stopwords(x) for x in df["plot"]]

    # Import label encoder 
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df["encoded_rating"] = label_encoder.fit_transform(df["rating"])
    
    #genre
    genres = pd.read_csv("https://data-bootcamp-ztc.s3.amazonaws.com/parsed_genres_table.csv")  # one hot encoding genres
    genres.set_index("name", inplace = True)
    genres = genres.drop(columns = ["genre_kaggle", "genres_omdb"])
    clean_df = pd.merge(df, genres, how = "inner", on = "name")

    #split dataset
    from sklearn.model_selection import train_test_split
    X = clean_df.drop(columns = ["rating","encoded_rating"])
    y = clean_df["encoded_rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # create tfidf vector
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    cv = HashingVectorizer().fit(X_train["plot"]) #hasher
    X_train_counts = cv.transform(X_train["plot"])
    X_test_counts = cv.transform(X_test["plot"])
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts) #transfromer
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    X_test_tfidf = tf_transformer.transform(X_test_counts)

    #scale plot len    
    from sklearn.preprocessing import MinMaxScaler
    plot_len_scaler = MinMaxScaler().fit(X_train["plot_len"].values.reshape(-1,1))
    X_train["plot_len"] = plot_len_scaler.transform(X_train["plot_len"].values.reshape(-1,1))
    X_test["plot_len"] = plot_len_scaler.transform(X_test["plot_len"].values.reshape(-1,1))

    #create extra features matrix
    import numpy as np
    from scipy.sparse import csr_matrix
    X_train_extra_features_mat = csr_matrix(X_train.drop(columns=["plot"])) 
    X_test_extra_features_mat = csr_matrix(X_test.drop(columns=["plot"]))  

    #combined the matrices
    from scipy.sparse import hstack
    X_train_full = hstack([X_train_tfidf, X_train_extra_features_mat])
    X_test_full = hstack([X_test_tfidf, X_test_extra_features_mat])

    # #fit the model

    from sklearn.svm import LinearSVC 

    clf = LinearSVC(class_weight='balanced',verbose=2, random_state=42,max_iter=100000)
    # from sklearn.svm import SVC 
    # model = SVC(kernel='linear',  class_weight='balanced')
    clf.fit(X_train_full, y_train)


    """
    end of model
    """ 

    # # GET request
    # if request.method == 'GET':
    #     message = {'greeting':'Hello from Flask!'}
    #     return jsonify(message)  # serialize and use JSON headers


    # POST request
    if request.method == 'POST':
        print(request.get_json())  # parse as JSON
        data = request.get_json()

        if data["plot"]:
            plot = data["plot"]
            plot_len = len(plot)
            plot = remove_punct(plot)
            plot = remove_stopwords(plot)
            hashed_vec = cv.transform([plot])
            tfidf_vec = tf_transformer.transform(hashed_vec)
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
        
        
        from scipy.sparse import csr_matrix

        extra_features_mat = csr_matrix(extra_features_df)

        from scipy.sparse import hstack

        input_mat = hstack([tfidf_vec, extra_features_mat])

        
        print(X_train_full.shape)
        print(input_mat.shape)

        output = clf.predict(input_mat)

        print(int(output[0]))


        print("-----------------------------")
        print("RAN SUCCESSFUL")
        print("-----------------------------")

        return render_template("rating.html", predictions = int(output[0]))


if __name__ == '__main__':
    app.run(debug=True)
