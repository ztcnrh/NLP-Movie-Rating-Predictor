# Movie Rating Predictor :movie_camera: - An ML and NLP Project

*Last Updated: June 19, 2021*

**Highlevel**: This is an application with an embedded Machine Learning (ML) algorithm utilizing Natural Language Processing (NLP) techniques that predicts movie MPA rating with movie plot and genres as inputs.<br>
**View the deployed app (Heroku) here**: WIP...

*View our presentation deck [here](https://docs.google.com/presentation/d/1IeEMjEUwlET_dmpA2JHjVTv4bb1F3u98FgvrikCeNEA/edit?usp=sharing)*

<img src=image_highlights/general/readme_header_hollywood.jpeg width="80%" alt="A picture of Hollywood">

## Background

This was the third and last data analysis project in the curriculum of Penn Data Boot Camp, which was a 24-week intensive online program focused on gaining data-analytics-oriented programming skills in various technologies and domains including Python and Machine Learning. The goal of this project was to **find a problem worth solving, analyzing, and visualizing at the same time use Machine Learning in the context of all the technologies we learned up to the moment when this was created.**

Thus, we decided to build algorithms based on a movies dataset because we were all movie lovers. We believed there is value in predicting various KPIs in terms of the "performance" or "merit" of a new movie based on historic data because the insights could potentially provide production companies and directors some guidelines on how to customize and what to expect. To give a very specific scenario (which is completely based on conjecture), companies like Netflix who specialized on tailoring the right content to the right audience on their powerful digital platform but at the same time also engaged in original movie and TV show productions could have interests in algorithms that could predict popularity or revenue, if they had not already been doing this. The last reasons we landed on this theme was because the data was readily available for us to collect. There were various cleaned datasets online and user friendly APIs which allowed us to gather quality data easily while focusing on the machine learning aspect.

### Data Source

* Base source - Three decades of movies scraped from IMDb by creator [Daniel Grijalva](https://www.kaggle.com/danielgrijalvas) ([Kaggle](https://www.kaggle.com/danielgrijalvas/movies))
  * Content: 6820 movies (220 movies per year, 1986-2016). Fields include: budget, production company, country, director, genre, gross, name, MPA rating, runtime, IMDb score, IMDb votes, star, writer, release year.
* Complementary source - [OMDB API](https://www.omdbapi.com/)
  * Some data such as awards and nominations were not in the dataset and some did not meet our project scopes, for example the "genre" field consisted of only one genre per movie while in reality movies are catogorized into more than one genre. Having more of that data also reflects the content of the movies more realistically and completely. As a result, we performed API calls from OMDB API, collected and consolidated additional data for our purpose into the base dataset.

## Project Scope & Summaries

*As a group of three, our approach at the beginning was to each take on a machine learning model to predict a different outcome. Depending on the models' performances and meaningfulness, we would then decide which model to implement into our flask application and deploy. While predicting movie MPA rating was our final choice, we will give a quick summary on the other models as well.*

* **Predict movie box office**
  * Model: Multiple Linear Regression
  * Features: time related features (year and month to account for seasonality), genres, factual and general features such as IMDb scores, votes, and runtime, budget.
  * Evaluation: F1-Score on the testing data was ~59%. The features that had the most impact in box office performance are IMDb votes and movie budget demonstrated by the regression coefficients.
* **Predict movie oscar nomination**
  * Model: Logistic Regression
  * Features: movie plot, plot length, genres
  * Evaluation: Class weight balanced F1-Score on the testing data was ~77%. However, the fact that the set of oscar winners in the dataset was disproportionally small created unbalanced dataset, making it hard to predict oscar nominated movies.
* **Predict movie MPA rating (implemented)**
  * Models: Decision Tree & Random Forests, Deep Learning, **Linear SVM (chosen)**
  * Features: movie plot, plot length, genres
  * Evaluation: In a nutshell, we created more than one model for this promising initiative and finally decided to implement the Linear SVM classifier. It was the fastest and it produced the best overall scores. In addition to a 64% F1-Score, it also had a reasonable score predicting G-rated movies, one of the rarest rating category in our dataset. The challenge we faced was that other models either had lower scores or had similar scores but had trouble predicting G-rated movies due to the unbalanced dataset. While the data reflects reality because there are fewer G-rated movies made every year, one way to solve this issue was to collect specifically more G-rated movies and broaden the range to prior to 1986 or later than 2016. After discussion, we dismissed this proposal due to time limitation but we reduced the imbalance by also applying `class_weight=balanced` parameter to the model. *Below is a report and a visualization of the confusion matrix generated from the testing data with the sklearn LinearSVC classifier:*

<img src=image_highlights/rating_svc/final_model_confusion_matrix_linear_svc_with_genre_(DP).png width="45%" alt="Confusion Matrix Heatmap of the Deployed Model - Linear SVM"><img src=image_highlights/rating_svc/final_model_scores_with_genre_linear_svc_(DP).png width="50%" alt="Classification Report of the Deployed Model - Linear SVM">

## Methods & Approach

**Technical Diagram of the App**

<img src=image_highlights/general/technical_diagram.png width="75%" alt="Workflow Technical Diagram">

* Our application was built in flask.
* ETL (parsing and inflation adjustment), encoding, and data preprocessing including Natural Language Processing (with movie plots) were performed and Machine Learning models were fitted. Throughout the process, the vectorizer, scaler along with the ML classifier were saved/pickled to be utilized in our flask app.
* User inputs were collected from the web and sent to flask app for outcome predictions. The outcome was then sent back and rendered on the web.

## App Highlights

**Prediction Demo 1 - Toy Story 4** *(Medium Viewport)*

<img src=image_highlights/app_highlights/toy_story_4_prediction_(2019).png width="75%" alt="Rating prediction using Toy Story 4's plot and genres">

**Prediction Demo 2 - Blade Runner 2049** *(Large Viewport)*

<img src=image_highlights/app_highlights/blade_runner_2049_prediction_(2017).png width="85%" alt="Rating prediction using Toy Story 4's plot and genres">

**Prediction Demo 3 - No plot information provided** *(Mobile-Sized Viewport)*

Valid plot data is requried for the algorithm to predict outcome, an error message will pop up to prompt user to fill the form.

<img src=image_highlights/app_highlights/media_query_and_warning_message.png width="45%" alt="Rating prediction using Toy Story 4's plot and genres">

<hr>

## Opportunities for Next Steps

* While in the real world there are way fewer G-rated than the PGs- and R-rated movies so our dataset reflected reality accurately. However, given more time, one way we could improve our model's accuracy across the board is to collect more data and manually make the data more balanced among the different ratings.
* All the people related features were left out of the models because we did not have a way to make the human names meaningful to the algorithm. For future improvement, if there's a not so cumbersome way to manually group writers, director and actors into "tiers" based on their resume or merit, they could be potentially additional indicators (aside from IMDb votes and scores which account for slightly different factors) for movie box office performance or the chance to win or get nominated for oscars.

