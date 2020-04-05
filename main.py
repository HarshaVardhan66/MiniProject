import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from flask import Flask, render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import os

app = Flask('__main__')


def linear_regression(full_data, value):
    X = full_data[['Date']]
    y = full_data.Temp
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    result = lin_reg.predict([[value]])
    return result


def AutoRegression(series, value):
    X = series.values
    for i in range(2001, value + 1):
        model = AR(X)
        model_fit = model.fit()
        y = model_fit.predict(len(X), len(X))
        X = list(X)
        X.append(y)
        X = np.asarray(X)
    return X[len(X) - 1]


def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error


def evaluate_order(dataset):
    dataset = dataset.astype('float32')
    p_values = range(0, 12)
    d_values = range(0, 5)
    q_values = range(0, 12)
    best_score, best_cfg = float("inf"), (7, 0, 1)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    return best_cfg


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, i, interval=1):
    return i + history[-interval]


def ARIMA_model(full_data, year):
    x = full_data.Temp.values
    dif = difference(x, 1)

    arima_order = evaluate_order(full_data)
    model = ARIMA(dif, order=arima_order)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=abs(year - 2012))[0]
    history = [i for i in x]
    value = 0
    for i in forecast:
        value = inverse_difference(history, i, 1)
        history.append(value)
    return value


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/show')
def show():
    return render_template('show.html')


@app.route('/show_result', methods=['POST'])
def showResult():
    # uploading data from csv file
    data = read_csv('D:/Harsha/Python/FullMiniProject/Data/Cities-Filtered.csv')

    # requesting city name from form and checking its presence
    name = request.form['place']
    name = name.capitalize()
    if name == 'Chennai':
        name = 'Madras'
    if name == 'Mumbai':
        name = 'Bombay'
    city = [name]

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
        add = add / 12
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
    return render_template('show.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/predict_temp', methods=['POST'])
def predict_temp():
    # Getting the required Data
    data = read_csv('D:/Harsha/Python/FullMiniProject/Data/Cities-Filtered.csv')
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
    full_data = DataFrame.from_dict(data_dict)

    year = int(request.form['year'])
    value = 0
    if request.form['method'] == 'Linear Regression':
        value = linear_regression(full_data, year)
    elif request.form['method'] == 'Auto Regression':
        full_data.to_csv('rough.csv', index=0)
        series = read_csv('rough.csv', index_col=0)
        os.remove('rough.csv')
        value = AutoRegression(series, year)
    elif request.form['method'] == 'Arima model':
        value = ARIMA_model(full_data, year)
        value = ' ' + str(value)
    return render_template('predict_result.html', Value=str(value)[1:7])


if __name__ == '__main__':
    app.run(debug=True)
