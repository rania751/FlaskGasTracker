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
     


df_info = pd.read_csv('TESTDATA.csv')
datat=df_info.copy()
df_info.drop('current_block_number', axis=1, inplace=True)
data1 = df_info[['current_datetime','gas_price_Gwei']]
data1.columns = ['ds','y']
train_size = int(len(data1) * 0.95)
train_data,test_data = data1[:train_size], data1[train_size:]





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
    with open('prophet_model_test_h_1.pkl', 'rb') as f:
       model = pickle.load(f)

    data = request.json   
    nbr=data['nbrint']
    if (nbr>180):
        return jsonify({"error": "Request Header Fields Too Large"}), 431

    # Assuming you have a previous date as a string
    previous_date_str = train_data['ds'].iloc[-1]

    previous_date = datetime.datetime.strptime(previous_date_str, '%Y-%m-%d %H:%M:%S')
    # Get the current timestamp
    current_timestamp = datetime.datetime.now()

    # Calculate the time difference between the previous date and the current timestamp
    time_difference = current_timestamp - previous_date


    # Calculate the number of hours
    hours_difference = time_difference.total_seconds() / 3600

    nbrH=round(nbr+hours_difference)



    future = model.make_future_dataframe(periods=nbrH, freq='H')
    forecast_future = future[future['ds'] > current_timestamp]

    # Make predictions for the future timestamps
    forecast = model.predict(forecast_future )

    # Access the forecasted values for the next 3 hours
    forecast_next = forecast[[ 'ds','yhat']]

    predicted_prices = forecast_next['yhat'].values.tolist()
    dates =forecast_next['ds'].tolist()


    return jsonify({
        'nbr':nbr,
        'dates': dates,
        'predictions': predicted_prices})





@app.route('/predictm', methods=['POST'])
def predictm():
    with open('prophet_model_test_h_1.pkl', 'rb') as f:
       model = pickle.load(f)

    data = request.json   
    nbr=data['nbrintm']
    print(nbr)
    if (nbr>180):
        return jsonify({"error": "Request Header Fields Too Large"}), 431

    # Assuming you have a previous date as a string
    previous_date_str = train_data['ds'].iloc[-1]

    previous_date = datetime.datetime.strptime(previous_date_str, '%Y-%m-%d %H:%M:%S')
    # Get the current timestamp
    current_timestamp = datetime.datetime.now()

    # Calculate the time difference between the previous date and the current timestamp
    time_difference = current_timestamp - previous_date

    mins_difference = time_difference.total_seconds() / 60
    print(mins_difference)

    nbrM=round(nbr+mins_difference)

    futureM = model.make_future_dataframe(periods=nbrM, freq='1min')
    forecast_future_M = futureM[futureM['ds'] > current_timestamp]

    forecast_M = model.predict(forecast_future_M )

    forecast_next_M = forecast_M[[ 'ds','yhat']]
   
    predicted_prices_M = forecast_next_M['yhat'].values.tolist()
    dates_M =forecast_next_M['ds'].tolist()

    return jsonify({
        'nbr':nbr,
        'dates_M': dates_M,
        'predictions_M': predicted_prices_M,
      
        })



if __name__ == "__main__":
    app.run(debug=True)