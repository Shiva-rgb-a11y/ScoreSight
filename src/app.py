from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get and validate form data
            reading_score = request.form.get("reading_score")
            writing_score = request.form.get("writing_score")

            # Convert to integers safely
            reading_score = int(reading_score) if reading_score and reading_score.isdigit() else 0
            writing_score = int(writing_score) if writing_score and writing_score.isdigit() else 0

            # Create data object
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("race_ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pred_df)

            # Predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template("home.html", results=results[0])

        except Exception as e:
            return render_template("home.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
