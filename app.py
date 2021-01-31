from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from splinter import Browser
from bs4 import BeautifulSoup as bs
import time
import csv
import datetime, time
import requests
import string
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from pathlib import Path

import warnings
warnings.simplefilter('ignore', FutureWarning)
from numpy.random import seed
seed(1)
#from sql_keys import username, password

# Create an instance of Flask
app = Flask(__name__)

# Use PyMongo to establish Mongo connection
#mongo = PyMongo(app, uri="mongodb://localhost:27017/web_scrapping_challenge_db")

rds_connection_string = "postgres:admin@localhost:5432/final_project_stocks"
#rds_connection_string = "postgres:admin@localhost:5432/final_project_stocks"
#<insert password>@localhost:5432/customer_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

#conn_url = 'postgres://slcslzlanikhqj:b106eda2173b6ce2c35f34626fb87eafcfd0f52e9ce55f36d776827fef375f71@ec2-52-3-4-232.compute-1.amazonaws.com:5432/ddv6vu8jpdbjns'
#engine = create_engine(conn_url)


#conn_url = 'postgres://wnhlndefflhtpu:d5f994af42137d89ab637af376441407d47cfbe163426ec823e3bf602599ee7c@ec2-54-163-215-125.compute-1.amazonaws.com:5432/dvfh8o0788t9q'
#engine = create_engine(conn_url)

#postgres://wnhlndefflhtpu:d5f994af42137d89ab637af376441407d47cfbe163426ec823e3bf602599ee7c@ec2-54-163-215-125.compute-1.amazonaws.com:5432/dvfh8o0788t9q


# Route to render index.html template using data from Mongo
@app.route("/")
def home():

    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!
    session = Session(engine)
    stocks = session.execute("select * from stocks ")
    #country = session.execute(" select country, country_code from country ")
    #return render_template("index.html", listings=listings)
    # Return template and data
  
    resdata = [{
  
    }
    ]

    responsedata = { 'respdata': resdata
    }
    session.close()
    return render_template("index.html", stocks=stocks, responsedata=responsedata, 
            init_page="initpage")


# Route to render index.html template using data from Mongo
@app.route("/kclass")
def kclass():

    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!
    session = Session(engine)
    stocks = session.execute("select * from stocks ")
    #return render_template("index.html", listings=listings)
    # Return template and data
  
    resdata = [{
  
    }
    ]

    responsedata = { 'respdata': resdata
    }
    session.close()
    return render_template("kclassification.html", stocks=stocks, responsedata=responsedata, 
            init_page="initpage")

#
#@app.route("/api/v1.0/<startdt>/<enddt>")
#def startEndDate(startdt, enddt):

# Route to render index.html template using data from Mongo
@app.route("/<st>")
def get_stocks(st):


    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!
    
    session = Session(engine)
    stocks = session.execute("select * from stocks ")
    #return render_template("index.html", listings=listings)
    # Return template and data

  
    resdata = [{
  
    }
    ]

    responsedata = { 'respdata': resdata
    }
    session.close()

    print('Hello this is test')
    df = pd.read_csv("static/data/"+st+".csv")
    # Drop the null columns where all values are null
    df = df.dropna(axis='columns', how='all')
    # Drop the null rows
    # This is for the MinMax Linear Regression model
    print(df.head())
    df = df.dropna()
    print(df.head())
    y = df["Open"].values.reshape(-1, 1)
    diff = df['Close']-df["Open"]
    diff_locations = []
    for i in diff:
        if (i <0):
            diff_locations.append(0)
        else:
            diff_locations.append(1)
    df['diff'] = pd.DataFrame(diff_locations)
    #X = df[['High', 'Low', 'Close', 'Volume','diff']]
    X = df[['High', 'Low', 'Close', 'Volume','diff']]
    print(X)
    print(y)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_minmax = MinMaxScaler().fit(X_train)
    y_minmax = MinMaxScaler().fit(y_train)

    X_train_minmax = X_minmax.transform(X_train)
    X_test_minmax = X_minmax.transform(X_test)
    y_train_minmax = y_minmax.transform(y_train)
    y_test_minmax = y_minmax.transform(y_test)
    model2 = LinearRegression()
    model2.fit(X_train_minmax, y_train_minmax)
    print(f"Testing Data Score: {model2.score(X_test_minmax, y_test_minmax)}")
    minmax_predict=model2.score(X_test_minmax, y_test_minmax)
    print(minmax_predict)

    #This is standard scalar transformation
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    predictions = model.predict(X_test_scaled)
    scallar_MSE = mean_squared_error(y_test_scaled, predictions)
    scallar_r2 = model.score(X_test_scaled, y_test_scaled)
    plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
    plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="orange", label="Testing Data")
    #plt.legend()
    plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
    plt.title("Residual Plot")
    #plt.show()
    pwd = os.getcwd()
    print(pwd)
    #p = Path(os.getcwd()+"\static\images")
    plt.savefig("static/images/"+st+".png")
    f = open("static/images/"+st+".png")
    plt.close()
    f.close()

    #Lasso model
    ### BEGIN SOLUTION
    lasso = Lasso(alpha=.01).fit(X_train_scaled, y_train_scaled)

    lasso_predictions = lasso.predict(X_test_scaled)

    lasso_MSE = mean_squared_error(y_test_scaled, lasso_predictions)
    lasso_r2 = lasso.score(X_test_scaled, y_test_scaled)
    ### END SOLUTION

    print(f"Lasso MSE: {lasso_MSE}, R2: {lasso_r2}")

    #Ridge model
    ridgeVal = Ridge(alpha=.01).fit(X_train_scaled, y_train_scaled)

    ridge_predictions = ridgeVal.predict(X_test_scaled)

    ridge_MSE = mean_squared_error(y_test_scaled, ridge_predictions)
    ridge_r2 = ridgeVal.score(X_test_scaled, y_test_scaled)
    print(f"ridge MSE: {ridge_MSE}, R2: {ridge_r2}")

    #elasticNet
    elasticnet = ElasticNet(alpha=.01).fit(X_train_scaled, y_train_scaled)

    elasticnet_predictions = elasticnet.predict(X_test_scaled)

    elasticnet_MSE = mean_squared_error(y_test_scaled, elasticnet_predictions)
    elasticnet_r2 = elasticnet.score(X_test_scaled, y_test_scaled)
    print(f"elasticnet MSE: {elasticnet_MSE}, R2: {elasticnet_r2}")

    fig1 = plt.figure(figsize=(12, 6))
    axes1 = fig1.add_subplot(1, 2, 1)
    axes2 = fig1.add_subplot(1, 2, 2)

    axes1.set_title("Original Data")
    axes2.set_title("Scaled Data")

    maxx = X_train["High"].max()
    maxy = y_train.max()
    axes1.set_xlim(-maxx + 1, maxx + 1)
    axes1.set_ylim(-maxy + 1, maxy + 1)

    axes2.set_xlim(-2, 2)
    axes2.set_ylim(-2, 2)
    set_axes(axes1)
    set_axes(axes2)

    axes1.scatter(X_train["High"], y_train)
    axes2.scatter(X_train_scaled[:,0], y_train_scaled[:])
    
    p = Path(os.getcwd()+"/static/images")
    #q = p / "axes2"+st+".png"
    #if (q.exists()):
    fig1.savefig("static/images/axes2"+st+".png")
    f = open("static/images/axes2"+st+".png")
    plt.close()
    f.close()
    #else:
    #    fig1.savefig("static/images/axes2"+st+".png")
    #    plt.close()



    
    return render_template("indexStocks.html", stocks=stocks, responsedata=responsedata, init_page="initpage", sel_stk=st, 
    minmax_predict=minmax_predict,
    scallar_MSE=scallar_MSE, scallar_r2=scallar_r2,
    lasso_MSE=lasso_MSE, lasso_r2=lasso_r2,
    ridge_MSE=ridge_MSE, ridge_r2=ridge_r2,
    elasticnet_MSE=elasticnet_MSE, elasticnet_r2=elasticnet_r2)
    

#
#@app.route("/api/v1.0/<startdt>/<enddt>")
#def startEndDate(startdt, enddt):

# Route to render index.html template using data from Mongo
@app.route("/upload/<st>")
def upload_get_stocks(st):


    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!

    #cr = csv.reader(open("https://query1.finance.yahoo.com/v7/finance/download/"+st+"?period1=1454112000&period2=1611964800&interval=1d&events=history&includeAdjustedClose=true","rb"))
 

    #data = pd.read_csv('https://example.com/passkey=wedsmdjsjmdd')

    #df = pd.read_csv("static/data/"+st+".csv")

    #with open("static/data/"+st+".csv", "wt") as fp:
    #    writer = csv.writer(fp)
    #    # writer.writerow(["your", "header", "foo"])  # write header
    #    writer.writerows(data)



    #dateval = datetime.date.strtime("%D")
    #print(dateval)
    session = Session(engine)
    stock = session.execute("select * from stocks where symbol='"+st+"'")
    #return render_template("index.html", listings=listings)
    # Return template and data
    

    if (stock.rowcount  == 0):
        data = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/"+st+"?period1=1454112000&period2=1611964800&interval=1d&events=history&includeAdjustedClose=true", sep=',')

        data.to_csv("static/data/"+st+".csv", index=False, header=True)
        
        print(data)
        session.execute("INSERT INTO stocks VALUES ('"+st+"', '"+st+" Corp')")
        session.execute("commit")



    stocks = session.execute("select * from stocks")


    resdata = [{
  
    }
    ]

    responsedata = { 'respdata': resdata
    }
    session.close()

    print('Hello this is test')
    data = pd.read_csv("static/data/"+st+".csv")
    df = data
    # Drop the null columns where all values are null
    df = df.dropna(axis='columns', how='all')
    # Drop the null rows
    # This is for the MinMax Linear Regression model
    print(df.head())
    df = df.dropna()
    print(df.head())
    y = df["Open"].values.reshape(-1, 1)
    diff = df['Close']-df["Open"]
    diff_locations = []
    for i in diff:
        if (i <0):
            diff_locations.append(0)
        else:
            diff_locations.append(1)
    df['diff'] = pd.DataFrame(diff_locations)
    #X = df[['High', 'Low', 'Close', 'Volume','diff']]
    X = df[['High', 'Low', 'Close', 'Volume','diff']]
    print(X)
    print(y)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_minmax = MinMaxScaler().fit(X_train)
    y_minmax = MinMaxScaler().fit(y_train)

    X_train_minmax = X_minmax.transform(X_train)
    X_test_minmax = X_minmax.transform(X_test)
    y_train_minmax = y_minmax.transform(y_train)
    y_test_minmax = y_minmax.transform(y_test)
    model2 = LinearRegression()
    model2.fit(X_train_minmax, y_train_minmax)
    print(f"Testing Data Score: {model2.score(X_test_minmax, y_test_minmax)}")
    minmax_predict=model2.score(X_test_minmax, y_test_minmax)
    print(minmax_predict)

    #This is standard scalar transformation
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    predictions = model.predict(X_test_scaled)
    scallar_MSE = mean_squared_error(y_test_scaled, predictions)
    scallar_r2 = model.score(X_test_scaled, y_test_scaled)
    plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
    plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="orange", label="Testing Data")
    #plt.legend()
    plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
    plt.title("Residual Plot")
    #plt.show()
    pwd = os.getcwd()
    print(pwd)
    #p = Path(os.getcwd()+"\static\images")
    plt.savefig("static/images/"+st+".png")
    f = open("static/images/"+st+".png")
    plt.close()
    f.close()

    #Lasso model
    ### BEGIN SOLUTION
    lasso = Lasso(alpha=.01).fit(X_train_scaled, y_train_scaled)

    lasso_predictions = lasso.predict(X_test_scaled)

    lasso_MSE = mean_squared_error(y_test_scaled, lasso_predictions)
    lasso_r2 = lasso.score(X_test_scaled, y_test_scaled)
    ### END SOLUTION

    print(f"Lasso MSE: {lasso_MSE}, R2: {lasso_r2}")

    #Ridge model
    ridgeVal = Ridge(alpha=.01).fit(X_train_scaled, y_train_scaled)

    ridge_predictions = ridgeVal.predict(X_test_scaled)

    ridge_MSE = mean_squared_error(y_test_scaled, ridge_predictions)
    ridge_r2 = ridgeVal.score(X_test_scaled, y_test_scaled)
    print(f"ridge MSE: {ridge_MSE}, R2: {ridge_r2}")

    #elasticNet
    elasticnet = ElasticNet(alpha=.01).fit(X_train_scaled, y_train_scaled)

    elasticnet_predictions = elasticnet.predict(X_test_scaled)

    elasticnet_MSE = mean_squared_error(y_test_scaled, elasticnet_predictions)
    elasticnet_r2 = elasticnet.score(X_test_scaled, y_test_scaled)
    print(f"elasticnet MSE: {elasticnet_MSE}, R2: {elasticnet_r2}")

    fig1 = plt.figure(figsize=(12, 6))
    axes1 = fig1.add_subplot(1, 2, 1)
    axes2 = fig1.add_subplot(1, 2, 2)

    axes1.set_title("Original Data")
    axes2.set_title("Scaled Data")

    maxx = X_train["High"].max()
    maxy = y_train.max()
    axes1.set_xlim(-maxx + 1, maxx + 1)
    axes1.set_ylim(-maxy + 1, maxy + 1)

    axes2.set_xlim(-2, 2)
    axes2.set_ylim(-2, 2)
    set_axes(axes1)
    set_axes(axes2)

    axes1.scatter(X_train["High"], y_train)
    axes2.scatter(X_train_scaled[:,0], y_train_scaled[:])
    
    p = Path(os.getcwd()+"/static/images")
    #q = p / "axes2"+st+".png"
    #if (q.exists()):
    fig1.savefig("static/images/axes2"+st+".png")
    f = open("static/images/axes2"+st+".png")
    plt.close()
    f.close()
    #else:
    #    fig1.savefig("static/images/axes2"+st+".png")
    #    plt.close()



    
    return render_template("indexStocks.html", stocks=stocks, responsedata=responsedata, init_page="initpage", sel_stk=st, 
    minmax_predict=minmax_predict,
    scallar_MSE=scallar_MSE, scallar_r2=scallar_r2,
    lasso_MSE=lasso_MSE, lasso_r2=lasso_r2,
    ridge_MSE=ridge_MSE, ridge_r2=ridge_r2,
    elasticnet_MSE=elasticnet_MSE, elasticnet_r2=elasticnet_r2)
   

# Route to render index.html template using data from Mongo
@app.route("/kclassification/<st>")
def kclassification(st):

    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!
    session = Session(engine)
    stocks = session.execute("select * from stocks ")
    #return render_template("index.html", listings=listings)
    # Return template and data
  
    resdata = [{
  
    }
    ]

    responsedata = { 'respdata': resdata
    }
    session.close()

    print('Hello this is test')
    df = pd.read_csv("static/data/"+st+".csv")
    # Drop the null columns where all values are null
    df = df.dropna(axis='columns', how='all')
    # Drop the null rows
    # This is for the MinMax Linear Regression model
    print(df.head())
    df = df.dropna()
    print(df.head())
    
    diff = df['Close']-df["Open"]
    diff_locations = []
    for i in diff:
        if (i <0):
            diff_locations.append(0)
        else:
            diff_locations.append(1)
    df['diff'] = pd.DataFrame(diff_locations)
    #X = df[['High', 'Low', 'Close', 'Volume','diff']]
    X = df[['Open','High', 'Low', 'Close', 'Volume']]
    y = df["diff"]
    print(X)
    print(y)
    print(st)
    print(X.shape, y.shape)
    #target_names = ["negative", "positive"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    train_scores = []
    test_scores = []

    for k in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_score = knn.score(X_train_scaled, y_train)
        test_score = knn.score(X_test_scaled, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
   
    plt.plot(range(1, 20, 2), train_scores, marker='o')
    plt.plot(range(1, 20, 2), test_scores, marker="x")
    plt.xlabel("k neighbors")
    plt.ylabel("Testing accuracy Score")


    p = Path(os.getcwd()+"/static/images")
    #q = p / "kclass"+st+".png"
    #if (q.exists()):
    plt.savefig("static/images/kclass"+st+".png")
    plt.close()
    f = open("static/images/kclass"+st+".png")
    f.close()
    #else:
    #    plt.savefig("static/images/kclass"+st+".png")
    #    plt.close()
    
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_scaled, y_train)
    print('k=13 Test Acc: %.3f' % knn.score(X_test_scaled, y_test))
    knn_score = knn.score(X_test_scaled, y_test)



    return render_template("kclassStocks.html", stocks=stocks, responsedata=responsedata, sel_stk_kc=st, 
            init_page="initpage",knn_score=knn_score)

def set_axes(ax):
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

if __name__ == "__main__":
    app.run(debug=True)
