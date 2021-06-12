# Dependencies
import nltk
import pandas as pd
import numpy as np
from os import environ
from flask import Flask, render_template, redirect, jsonify, request
# import sqlalchemy
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
# from sqlalchemy.orm import Session
# from sqlalchemy.ext.automap import automap_base
# from sqlalchemy import create_engine, func, inspect, or_


app = Flask(__name__)

# # Connect to Heroku Postgres if running on server or sqlite if running locally
# app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL', '') or "sqlite:///static/data/data_all.sqlite"
# # Connect to Heroku Config Vars if running on server or grab api key var from config.py if running locally
# app.config['MAP_API_KEY'] = environ.get('API_KEY', '') or LOCAL_API_KEY

# # Remove tracking modifications
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'

# db = SQLAlchemy(app)
# ma = Marshmallow(app)

# from .models import Parking, Weather
# from .models import ppa_schema

# database_path = '../flask_app/static/data/data_all.sqlite'
# engine = create_engine(f'sqlite:///{database_path}')
# conn = engine.connect()

# # Reflect an existing database into a new model
# Base = automap_base()
# # Reflect the tables
# Base.prepare(engine, reflect=True)

# # View all of the classes that automap found
# print(Base.classes.keys())

# # Save references to each table
# # Movies = Base.classes.movies
# # --------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------


# # Routes to retrieve data
# # ---------------------------------
# # ---------------------------------
# @app.route('/api/data')
# def get_data():
#     # Create our session (link) from Python to the DB
#     session = Session(engine)

#     # Query
#     # ----- Data for specific violation types -----
#     data = session.query()

#     # ------------------------
#     # Session ends, all queries completed
#     session.close()

#     return jsonify(data)



# Routes to render templates
# ---------------------------------
# ---------------------------------

# Home Route
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods =['GET','POST'])
def predict():

       # GET request
    if request.method == 'GET':
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers
    # POST request
    if request.method == 'POST':
        
        
        print(request.get_json())  # parse as JSON
        data = request.get_json()


        """        
        preprocessing pipeline
        """        

        # #load in original training data
        movies_df = pd.read_csv("https://data-bootcamp-ztc.s3.amazonaws.com/movies_complete_cleaned.csv")
        df = movies_df[["name", "plot"]]
        df.set_index("name",inplace = True)

        # df = df.dropna(axis='index', subset=['plot'])
        df = df.dropna(axis='index', subset=['plot'])

         # #generate plot_len
        df["plot_len"] = [len(x) for x in df["plot"]]

        # ##remove puctuation 
        import string
        nltk.download('punkt')
        def remove_punct(text):
            table = str.maketrans("", "", string.punctuation)
            return text.translate(table)
        df["plot"] = [remove_punct(x) for x in df["plot"]]

        # ## remove stop words
        
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english"))
        def remove_stopwords(text):
            text = [word.lower() for word in text.split() if word.lower() not in stop]

            return " ".join(text)
        df["plot"] = [remove_stopwords(x) for x in df["plot"]]

        # # Lemmatization
        # from nltk.stem import WordNetLemmatizer
        # nltk.download('wordnet')
        # wordnet_lemmatizer = WordNetLemmatizer()

        # def stemming(stopwords_removed):
        #     text = [wordnet_lemmatizer.lemmatize(word) for word in stopwords_removed]
        #     return text

        # df["plot"] = [stemming(x) for x in df["plot"]]
      
       
        # ## split dataset
        from sklearn.model_selection import train_test_split
        X = df[["plot_len", "plot"]]
        y = df.index #placeholder

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # #  # vectoriser and transformer
        # from sklearn.feature_extraction.text import HashingVectorizer
        # from sklearn.feature_extraction.text import TfidfTransformer
       
        # cv = HashingVectorizer().fit(X_train["plot"])
        # X_train_counts = cv.transform(X_train["plot"])
        # tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)


        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer().fit(X_train["plot"])

        # # Scaler for plot_len
        from sklearn.preprocessing import MinMaxScaler
        plot_len_scaler = MinMaxScaler().fit(X_train["plot_len"].values.reshape(-1,1))

        # """
        # end of preporocessing pipeline
        # """


        # print(data["plot"])

        if data["plot"]:

            plot = data["plot"]
            plot = remove_punct(plot)
            plot = remove_stopwords(plot)
            plot_len = len(plot)
            # hashed_vec = cv.transform([plot])
            # tfidf_vec = tf_transformer.transform(hashed_vec)
            tfidf_vec = vectorizer.transform([plot])
            plot_len_scaled = plot_len_scaler.transform(np.array(plot_len).reshape(-1,1))

        # print(tfidf_vec.shape)
        # print(tfidf_vec)
        # print(tfidf_vec.todense())
        # print(type(tfidf_vec))

     
        # print(type(plot_len_scaled))
        # # print(len(plot_len_scaled))
        # print(plot_len_scaled[0][0])

        extra_features_df = pd.DataFrame({
            "action": data["action"],
            # "adventure": data["adventure"],
            # "fantasy"	: data["fantasy"],
            # "sci-fi"	: data["sci-fi"],            
            # "crime"	: data["crime"],
            # "drama"	: data["drama"],
            # "history" : data["history"],
            # "comedy"	: data["comedy"],
            # "biography"	: data["biography"],
            # "romance"	: data["romance"],
            # "horror"	: data["horror"],
            # "thriller"	: data["thriller"],
            # "war"	: data["war"],
            # "animation"	: data["animation"],
            # "family"	: data["family"],
            # "sport" : data["family"],
            # "music" : data["music"],
            # "mystery"	: data["mystery"],
            # "short" : data["short"],
            # "western" : data["western"],
            # "musical"	: data["musical"],
            # "documentary" : data["documentary"],
            # "film-noir" : data["film-noir"],
            # "adult" : data["adult"],
            "plot_len" : plot_len_scaled[0][0]
            }, index = [0])
        
        
        # print(input_df)

        from scipy.sparse import csr_matrix

        extra_features_mat = csr_matrix(extra_features_df)

        from scipy.sparse import hstack

        input_mat = hstack([extra_features_mat, tfidf_vec])

        print(input_mat.shape)


        print("-----------------------------")
        print("RAN SUCCESSFUL")
        print("-----------------------------")
        return 'Sucesss', 200


if __name__ == '__main__':
    app.run(debug=True)
