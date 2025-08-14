from flask import Flask, request, render_template

import pickle
import json
import numpy as np

app = Flask(__name__)                                                # object app(__name__)   magical methods

# Load model and schema(best)
with open("ev_model.pkl", 'rb') as file:
    model = pickle.load(file)  # we store our model in model variable

with open("model_schema.json", "r") as file:
    schema = json.load(file)

features = schema['features']  # we access all feature here to show user


# home page
@app.route("/")                                                      # decorators home page
def home():
    return render_template("index.html")
    # return "hello python"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form                                           # it is use to take input from user
        input_list = []
        input_list.append(float(data['Age']))                         # by using data we take age as as input and append in input list
        input_list.append(float(data['Income']))
        input_list.append(float(data['Vehicle_Budget']))
        input_list.append(float(data['Driving_Habits']))
        input_list.append(float(data['Environmental_Concern']))

        cities = ['City_Delhi', 'City_Hyderabad', 'City_Mumbai', 'City_Pune']
        selected_city = data['City']
        for city in cities:
            input_list.append(1 if selected_city in city else 0)

        input_array = np.array(input_list).reshape(1, -1)
        prediction = model.predict(input_array)[0]  # 1 or 0
        result = "Will buy EV" if prediction == 1 else "Will Not Buy EV"
        return render_template("index.html", prediction_text=f"prediction :{result}")


    except Exception as e:
        print(f"Error: {e}")  # This will show the real error in your terminal
        return render_template("index.html", prediction_text=f"Error: {e}")



if __name__ == "__main__":
    app.run(debug=True)