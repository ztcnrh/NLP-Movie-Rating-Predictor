-- Create all tables

CREATE TABLE all_movies (
  name VARCHAR PRIMARY KEY NOT NULL,
  production VARCHAR,
  director VARCHAR,
  runtime INT,
  released VARCHAR,
  year INT,
  month INT,
  country_kaggle VARCHAR,
  country_omdb VARCHAR,
  star_kaggle VARCHAR,
  actors_omdb VARCHAR,
  writer_kaggle VARCHAR,
  writers_omdb VARCHAR,
  language_omdb VARCHAR,
  plot VARCHAR,
  awards VARCHAR,
  score_imdb FLOAT,
  votes_imdb FLOAT,
  score_metacritic FLOAT,
  budget FLOAT,
  genre_kaggle VARCHAR,
  gross FLOAT,
  genres_omdb VARCHAR,
  rating VARCHAR
);

CREATE TABLE awards (
  name VARCHAR PRIMARY KEY NOT NULL,
  awards VARCHAR,
  oscar_wins_and_nominations INT,
  other_wins INT,
  other_nominations INT,
  total_awards_and_nominations INT
);

CREATE TABLE genres (
  name VARCHAR PRIMARY KEY NOT NULL,
  genre_kaggle VARCHAR,
  genres_omdb VARCHAR,
  action INT,
  adventure INT,
  fantasy INT,
  sci_fi INT,
  crime INT,
  drama INT,
  history INT,
  comedy INT,
  biography INT,
  romance INT,
  horror INT,
  thriller INT,
  war INT,
  animation INT,
  family INT,
  sport INT,
  music INT,
  mystery INT,
  short INT,
  western INT,
  musical INT,
  documentary INT,
  film_noir INT,
  adult INT
);

CREATE TABLE months (
  name VARCHAR PRIMARY KEY NOT NULL,
  month INT,
  Jan INT,
  Feb INT,
  Mar INT,
  Apr INT,
  May INT,
  Jun INT,
  Jul INT,
  Aug INT,
  Sep INT,
  Oct INT,
  Nov INT,
  Dec INT,
  spring INT,
  summer INT,
  fall INT,
  winter INT
);

CREATE TABLE adjusted (
  name VARCHAR PRIMARY KEY NOT NULL,
  year INT,
  budget FLOAT,
  gross FLOAT,
  adjusted_budget FLOAT,
  adjusted_gross FLOAT
);
