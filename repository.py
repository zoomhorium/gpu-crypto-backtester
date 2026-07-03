import os

import requests
from dotenv import load_dotenv
import urllib.parse
import hmac
import hashlib
import time
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "apikey.env"))

api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")


def get_historical_data(symbol,limit ,interval="1h"):
    data = requests.get(f"https://demo-fapi.binance.com/fapi/v1/continuousKlines?pair={symbol}&interval={interval}&limit={limit}&contractType=PERPETUAL")
    chartdata = data.json()
    return chartdata
def get_historical_data_trainning(symbol, interval="15m",startTime=None,endTime=None):
    data = requests.get(f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={startTime}&endTime={endTime}&limit=1000")
    chartdata = data.json()
    points = []
    for candle in chartdata:
        point = {
            "openTime": candle[0],
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
            "closeTime": candle[6],
        }
        points.append(point)
    return points
def get_ballance(): 
    params = {"timestamp": int(time.time()*1000)}
    params["signature"]=get_signature(params,secret_key)
    accaunt= requests.get("https://demo-fapi.binance.com/fapi/v2/balance", params=params,headers = {"X-MBX-APIKEY": api_key})
    for item in accaunt.json():
        if item["asset"] == "USDT":
            balance = float(item["balance"])
        
    
    return balance
def open_position(symbol,side,quantity,trailingDelta,tProfit):
    params = {"symbol":symbol,
              "side":side,
              "type":"MARKET",
              "quantity":quantity,
              "trailingDelta":trailingDelta,
              }
    params["timestamp"] = int(time.time()*1000)
    params["signature"] = get_signature(params, secret_key)
    order=requests.post(f"https://demo-fapi.binance.com/fapi/v1/order",params=params,headers = {"X-MBX-APIKEY": api_key})
    return order
def close_position(symbol,side,quantity):
    
    params ={
        
        "quantity":quantity,
        "type":"MARKET",
        "side":side,
        "symbol":symbol,
        "reduceOnly":"true"
    }
    params["timestamp"] = int(time.time()*1000)
    params["signature"] = get_signature(params, secret_key)
    data=requests.post(f"https://demo-fapi.binance.com/fapi/v1/order",params=params,headers = {"X-MBX-APIKEY": api_key})

def get_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(str(secret), 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac
