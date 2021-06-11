# Dependencies
from os import environ
from flask import Flask, render_template, redirect, jsonify, request
import sqlalchemy
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, func, inspect, or_


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
# Movies = Base.classes.movies
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# Routes to retrieve data
# ---------------------------------
# ---------------------------------
@app.route('/api/data')
def get_data():
    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Query
    # ----- Data for specific violation types -----
    data = session.query()

    # ------------------------
    # Session ends, all queries completed
    session.close()

    return jsonify(data)



# Routes to render templates
# ---------------------------------
# ---------------------------------

# Home Route
@app.route('/')
def index():
    return render_template('index.html')







if __name__ == '__main__':
    app.run(debug=True)
