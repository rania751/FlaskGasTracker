import requests
import json
import time
import datetime
import urllib3
import csv
import pandas as pd

gas_tracker_df = pd.DataFrame()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

api_key = "XNN6JYB3CUK5Z7BUP83SQ7Z53K1ID9P1HJ"

# Etherscan API endpoint
GasTrackerOracle_URL = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}"
BlockNumber_URL = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={api_key}"
latestTx = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag=latest&boolean=true&apikey={api_key}"
GasPrice_URL = f"https://api.etherscan.io/api?module=proxy&action=eth_gasPrice&apikey={api_key}"

# Load existing data from the CSV file
try:
    gas_tracker_df = pd.read_csv("gastracker.csv")
except FileNotFoundError:
    gas_tracker_df = pd.DataFrame()

# Set the number of iterations
nb_iterations = 500000000000000000000000000

# Loop for the specified number of iterationss
for i in range(nb_iterations):
#i=0
#while(True):
   
    print(f"\n Iteration {i+1}")

    current_timestamp = int(time.time())
    current_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp))
    print("TimeStamp:", current_datetime)
    
    response = requests.get(GasPrice_URL, verify=False)
    data = json.loads(response.text)
    # Extract gas price in Gwei
    gas_price = int(data["result"], 16) / 10**9

    response = requests.get(GasTrackerOracle_URL, verify=False)
    current_gas_price = response.json()



    response = requests.get(BlockNumber_URL, verify=False)
    block_number = int(response.json()['result'], 16)
    current_block_number=block_number+1


    # print(f"Latest BlockNumber:", block_number) 
    print("current_block_number", current_block_number )

    print(f"Current Gas Price in Gwei: {gas_price}")
    
    my_dict={
        "current_datetime": current_datetime,
        "current_block_number":current_block_number,
        "gas_price_Gwei": gas_price,

    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(my_dict, index=[0])
    
    # Concatenate the new DataFrame with the existing DataFrame
    gas_tracker_df = pd.concat([gas_tracker_df, df])
    
    # Save the combined DataFrame back to the CSV file
    gas_tracker_df.to_csv("gastracker.csv", index=False)

    # Wait for 6 seconds
    time.sleep(6)
    #i+=1