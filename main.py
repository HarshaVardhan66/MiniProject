import matplotlib.pyplot as plt
from pandas import *
from flask import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR
import os

app = Flask('__main__')


def linear_regression(full_data, value) :
    X = full_data[['Date']]
    y = full_data.Temp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    result = lin_reg.predict([[value]])
    return result


def AutoRegression(series, value):
    X = series.values
    for i in range(2001, value+1) :
        model = AR(X)
        model_fit = model.fit()
        y = model_fit.predict(len(X),len(X))
        X = list(X)
        X.append(y)
        X = np.asarray(X)
    return X[len(X)-1]


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/show')
def show():
    return render_template('show.html')


@app.route('/showResult', methods=['POST'])
def showResult():
    # uploading data from csv file
    data = read_csv('D:/Harsha/Python/MiniProject/Data/Cities-Filtered.csv')

    # requesting city name from form and checking its presence
    name = request.form['place']
    name = name.capitalize()
    if name == 'Chennai':
        name = 'Madras'
    if name == 'Mumbai':
        name = 'Bombay'
    city = [name]
    if name not in set(data.City):
        return "<h1><Center>No such city Found"

    # after finding city in data, listing out the dates excluding 2013
    data = data[data.City.isin(city)]
    dates = list(set(data['dt']))
    dates.sort()

    # requesting start year and end year from form
    start = request.form['start']
    end = request.form['end']
    temps = list(data['AverageTemperature'])
    dates = list(data['dt'])
    year = int(start)

    # getting required average temperatures to plot the appropriate graph
    data_dict = {}
    for i in range(dates.index(start + '-01-01'), dates.index(end + '-12-01'), 12):
        add = 0
        for j in range(0, 12):
            add = add + float(temps[i + j])
        add = add/12
        data_dict[year] = add
        year = year + 1

    # plotting graph for keys and values of data_dict
    x = list(data_dict.keys())
    y = list(data_dict.values())
    plt.plot(x, y)
    plt.xlabel('Years')
    plt.ylabel('Temperature')

    # saving the graph data into the file
    plt.show()
    return '<title>Result</title><Center><a href="http://localhost:5000/show">Try ' \
           'Again</a><br><br><a href="http://localhost:5000/">Home</a> '


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/predict_header')
def predict_header():
    return render_template('predict_header.html')


@app.route('/predict_temp', methods=['POST'])
def predict_temp():
    # Getting the required Data
    data = pd.read_csv(r'D:\Harsha\Python\MiniProject\Data\Cities-Filtered.csv')
    name = request.form['city']
    name = name.capitalize()
    if name == 'Chennai':
        name = 'Madras'
    if name == 'Mumbai':
        name = 'Bombay'
    city = [name]

    data = data[data['City'].isin(city)]
    data = data[['dt', 'AverageTemperature']]
    data = data.reset_index(drop=True)
    data = data.rename(columns={'dt': 'Date', 'AverageTemperature': 'Temp'})

    dates = list(set(data['Date']))
    dates.sort()
    temps = list(data['Temp'])

    # Extracting Fitting data
    start = '1901'
    end = '2000'
    year = int(start)
    data_dict = {'Date': [], 'Temp': []}
    for i in range(dates.index(start + '-01-01'), dates.index(end + '-12-01'), 12):
        add = 0
        for j in range(0, 12):
            add = add + float(temps[i + j])
        add = add / 12
        data_dict['Date'].append(year)
        data_dict['Temp'].append(add)
        year = year + 1
    full_data = pd.DataFrame.from_dict(data_dict)

    year = int(request.form['year'])

    if request.form['method'] == 'linear':
        value = linear_regression(full_data, year)
    elif request.form['method'] == 'ar':
        full_data.to_csv('rough.csv', index=0)
        series = read_csv('rough.csv', index_col=0)
        os.remove('rough.csv')
        value = AutoRegression(series, year)
    return "<center>Value is : " + str(value)[1:7]


if __name__ == '__main__':
    app.run(debug=True)
