from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS, cross_origin
from flask_session import Session
from config import ApplicationConfig
from models import db, User
from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import datetime
import pandas as pd
import requests
import json
import time
import datetime
import urllib3
import pandas as pd
import sklearn
import numpy as np
from fbprophet.plot import plot_components_plotly
from fbprophet.plot import plot_plotly

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


app = Flask(__name__)
app.config.from_object(ApplicationConfig)

bcrypt = Bcrypt(app)
CORS(app, supports_credentials=True)
server_session = Session(app)
db.init_app(app)

with app.app_context():
    db.create_all()




api_key = "XNN6JYB3CUK5Z7BUP83SQ7Z53K1ID9P1HJ"

# Etherscan API endpoint
GasTrackerOracle_URL = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}"
BlockNumber_URL = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={api_key}"
latestTx = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag=latest&boolean=true&apikey={api_key}"
GasPrice_URL = f"https://api.etherscan.io/api?module=proxy&action=eth_gasPrice&apikey={api_key}"


response = requests.get(GasPrice_URL, verify=False)
response1 = requests.get(GasTrackerOracle_URL, verify=False)
response2 = requests.get(BlockNumber_URL, verify=False)
     


def calculate_mean_squared_error(actual, predicted):
    squared_errors = np.square(actual - predicted)
    return np.mean(squared_errors)

# Function to calculate mean absolute error (MAE)
def calculate_mean_absolute_error(actual, predicted):
    absolute_errors = np.abs(actual - predicted)
    return np.mean(absolute_errors)

# Function to calculate root mean squared error (RMSE)
def calculate_root_mean_squared_error(actual, predicted):
    squared_errors = np.square(actual - predicted)
    mean_squared_error = np.mean(squared_errors)
    return np.sqrt(mean_squared_error)


def MAPE(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean([100*abs((actual[i]-forecast[i])/actual[i]) for i in range(len(forecast)-1)])

def metrics(df_train, forecast_train, df_test, forecast_test):
    forecast_train = forecast_train[forecast_train['yhat'].notna()]
    forecast_test = forecast_test[forecast_test['yhat'].notna()]

    MAPE_metric = pd.DataFrame(index=['MAPE'],
                               data={'Train': [MAPE(df_train['y'], forecast_train['yhat'])],
                                     'Test': [MAPE(df_test['y'], forecast_test['yhat'])]})

    
    return ([MAPE_metric])




df_info = pd.read_csv('SOFREC.csv')
datat=df_info.copy()
df_info.drop('current_block_number', axis=1, inplace=True)
data1 = df_info[['current_datetime','gas_price_Gwei']]
# Convert 'current_datetime' column to datetime format
data1['current_datetime'] = pd.to_datetime(data1['current_datetime'])

# Set the 'current_datetime' column as the DataFrame index
data1.set_index('current_datetime', inplace=True)

# Resample the DataFrame to one-minute intervals and take the first row of each minute
resampled_data = data1.resample('1Min').first()

resampled_data.reset_index(level=0, inplace=True)


# Assuming 'data' is your DataFrame with columns 'ds' and 'y'
resampled_data['current_datetime'] = pd.to_datetime(resampled_data['current_datetime'])  # Convert 'ds' column to datetime format

# Perform linear interpolation
resampled_data['y_interpolated'] = resampled_data['gas_price_Gwei'].interpolate(method='linear')


resampled_data['gas_price_Gwei']=resampled_data['y_interpolated']
resampled_data.drop('y_interpolated', axis=1, inplace=True)

resampled_data=resampled_data.reset_index(drop=True)


resampled_data.columns = ['ds','y']
train_size = int(len(resampled_data) * 0.9)


train_data,test_data = resampled_data[:train_size], resampled_data[train_size:]





@app.route("/@data", methods=["GET"])
def get_all_data():
    current_datetime=datat['current_datetime'].tolist()
    gas_price_Gwei=datat['gas_price_Gwei'].tolist()
    current_block_number=datat['current_block_number'].tolist()
    

    return jsonify({
        "current_datetime": current_datetime,
        "gas_price_Gwei": gas_price_Gwei,
         "current_block_number":current_block_number
    }) 




@app.route("/gasprice", methods=["GET"])
def get_gas_price():
   

    response = requests.get(GasPrice_URL, verify=False)
    response1 = requests.get(GasTrackerOracle_URL, verify=False)
    response2 = requests.get(BlockNumber_URL, verify=False)
        

    data = json.loads(response.text)
    gas_price = int(data["result"], 16) / 10**9
    current_gas_price = response1.json() 
    safe_gas_price = current_gas_price["result"]["SafeGasPrice"]
    propose_gas_price = current_gas_price["result"]["ProposeGasPrice"]
    fast_gas_price = current_gas_price["result"]["FastGasPrice"]
    BaseFee = current_gas_price["result"]["suggestBaseFee"]
    priority_safe = float(safe_gas_price)-float(BaseFee)
    priority_propose = float(propose_gas_price)-float(BaseFee)
    priority_fast = float(fast_gas_price)-float(BaseFee)
    block_number = int(response2.json()['result'], 16)
    current_block_number=block_number+1
    print(gas_price)
    return jsonify({
        "current_block_number":current_block_number,
        "gas_price": gas_price,
        "safe_gas_price": safe_gas_price,
        "ProposeGasPrice": propose_gas_price,
        "fast_gas_price": fast_gas_price,
        "priority_safe": priority_safe,
        "priority_propose": priority_propose,
        "priority_fast": priority_fast,
        "BaseFee": BaseFee,
    }) 


@app.route("/@me", methods=["GET"])
def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.filter_by(id=user_id).first()
    return jsonify({
        "id": user.id,
        "email": user.email,
         "username":user.username
    }) 

@app.route("/register", methods=["POST"])
def register_user():
    email = request.json["email"]
    password = request.json["password"]
    username= request.json["username"]

    user_exists = User.query.filter_by(email=email).first() is not None


    if user_exists:
        return jsonify({"error": "User already exists"}), 409
    

    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(username=username,email=email, password=hashed_password )
    db.session.add(new_user)
    db.session.commit()
    
    session["user_id"] = new_user.id

    return jsonify({
        "id": new_user.id,
        "email": new_user.email,
        "username":new_user.username
        
        
    })

@app.route("/login", methods=["POST"])
def login_user():
    email = request.json["email"]
    password = request.json["password"]

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"error": "Unauthorized"}), 401

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401
    
    session["user_id"] = user.id

    return jsonify({
        "id": user.id,
        "email": user.email
    })

@app.route("/logout", methods=["POST"])
def logout_user():
    session.pop("user_id")
    return "200"


@app.route('/predict', methods=['POST'])
def predict():
    with open('Bestmodel_04_mae_5.52.pkl', 'rb') as f:
       model = pickle.load(f)

    data = request.json   
    nbr=data['nbrint']

    if (nbr>180):
        return jsonify({"error": "Request Header Fields Too Large"}), 431

    # Assuming you have a previous date as a string
    previous_date_str = train_data['ds'].iloc[-1]
  
    # previous_date = datetime.datetime.strptime(previous_date_str, '%Y-%m-%d %H:%M:%S')
   
    # Get the current timestampQ
    current_timestamp = datetime.datetime.now()

    # Calculate the time difference between the previous date and the current timestamp
    time_difference = current_timestamp - previous_date_str

    # Calculate the number of hours
    hours_difference = time_difference.total_seconds() / 3600

    nbrH=round(nbr+hours_difference)
    print(nbrH)

    future = model.make_future_dataframe(periods=nbrH, freq='H')
    print(future)
    forecast_future = future[future['ds'] > current_timestamp]
    print(forecast_future)

    # Make predictions for the future timestamps
    forecast = model.predict(forecast_future )

    # Access the forecasted values for the next 3 hours
    forecast_next = forecast[[ 'ds','yhat']]
    print(forecast_next)

    predicted_prices = forecast_next['yhat'].values.tolist()
    print(predicted_prices)
    dates =forecast_next['ds'].tolist()

    return jsonify({
        'nbr':nbr,
        'dates': dates,
        'predictions': predicted_prices})





@app.route('/eval', methods=['GET'])
def eval():   
    with open('Bestmodel_04_mae_5.52.pkl', 'rb') as f:
       model = pickle.load(f)

    future = model.make_future_dataframe(periods=len(test_data), freq='1min')  # 6 hours of future data
    forecast = model.predict(future)
    forecast_next = forecast[['ds', 'yhat']]   
    forecast_data=forecast_next[:len(train_data)]
    forecast_test= forecast_next[len(train_data):]
    mse=calculate_mean_squared_error(test_data['y'], forecast_test['yhat'])
    mae=calculate_mean_absolute_error(test_data['y'], forecast_test['yhat'])
    rmse=calculate_root_mean_squared_error(test_data['y'], forecast_test['yhat'])
    mape_train=MAPE(train_data['y'], forecast_data['yhat'])
    mape_test=MAPE(test_data['y'], forecast_test['yhat'])
    testp=forecast_next[len(train_data):]['yhat'].values.tolist()
    # testp=forecast_next[:]['yhat'].values.tolist()
    testa=test_data[:]['y'].values.tolist()
    datestest =forecast_next[len(train_data):]['ds'].values.tolist()
    datestestall =forecast_next[:]['ds'].values.tolist()
    datestestrain=train_data[:]['ds'].values.tolist()
    fig1=plot_plotly(model ,forecast)
    fornext=forecast_next['yhat'].values.tolist()
    datta=resampled_data['y'].values.tolist()
    train=train_data['y'].values.tolist()
    
    print (len(fornext))
    print (len(datta))


    testp_data = {
        "x": datestest,
        "y": testp,
        "type": "scatter",
        "mode": "lines",
        "marker": {"color": "blue"},
        "name": "Test Prediction"
    }

    # Create the testa plot data
    testa_data = {
        "x": datestest,
        "y": testa,
        "type": "scatter",
        "mode": "lines",
        "marker": {"color": "green"},
        "name": "Actual Values"
    }

    plottest = [testp_data, testa_data]

    alldata = {
        "x": datestestall,
        "y": datta,
        "type": "scatter",
        "mode": "lines",
        "marker": {"color": "green"},

        "name": "Actual Prediction"
    }

    # Create the testa plot data
    fordata = {
        "x": datestestall,
        "y": fornext,
        "type": "scatter",
        "mode": "lines",
        "marker": {"color": "red"},
        "name": "Test Values"
    }
    traindata = {
        "x": datestestall,
        "y": train,
        "type": "scatter",
        "mode": "lines",
        "marker": {"color": "blue"},
        "name": "Test Values"
    }
    

    plotall = [alldata,fordata]
    plotfor = [fordata]
    # trainplott=[traindata,fordata]
    trainplott=[traindata,fordata]



    plot_data1=fig1.to_json()

    return jsonify({
        "mse":mse,
        "mae": mae,
        "rmse": rmse,
        "mape_train":mape_train,
        "mape_test":mape_test,
        "plot_data1":plot_data1,
        "plotall": plotall,
        "plotfor":plotfor,
        "trainplott":trainplott,
        "plottest": plottest,
        "testp":testp,
        "testa":testa,
        "datestest":datestest,
        
    }) 

if __name__ == "__main__":
    app.run(debug=True)