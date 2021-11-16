import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import pandas as pd
import math
import numpy_financial as npf


class LinRes(nn.Module):
    def __init__(self, input_shape):
        super(LinRes, self).__init__()
        self.layer_in = nn.Linear(input_shape, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64,1)

    
    def forward(self, x):
        x = torch.relu(self.layer_in(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = x.reshape(-1)
        return x
      
      
def spherical_dist(pos1, pos2, r=3958.75):
  pos1 = pos1 * np.pi / 180
  pos2 = pos2 * np.pi / 180
  cos_lat1 = np.cos(pos1[..., 0])
  cos_lat2 = np.cos(pos2[..., 0])
  cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
  cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
  return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))
  
def checking_nearest_university(loc, loc_u):
  dist = spherical_dist(loc, loc_u)
  shortest_dist = np.min(dist)
  nearest_u = np.argmin(dist)
  
  return shortest_dist, nearest_u

      
      
input_shape = 7
checkpoint_model = torch.load('nd/demand_model.pt')
model = LinRes(input_shape)
model.load_state_dict(checkpoint_model['model_state_dict'])

with open('nd/scaler.pickle', 'rb') as f:
  scaler = pickle.load(f)
  
universities = pd.read_csv('nd/dataset/ipt_johor.csv')

longitude, latitude = list(), list()
for longlat in universities.LongLat:
  lat, long = longlat.split(',')
  longitude.append(float(long))
  latitude.append(float(lat))
  
universities['latitude'] = latitude 
universities['longitude'] = longitude

location_ipt = universities[['latitude', 'longitude']].values

def predict(**kwargs):

    latitude, longitude = kwargs['latitude'], kwargs['longitude']
    location_fdc = np.array([latitude, longitude])
    nearest_ipt_distance = checking_nearest_university(location_fdc, location_ipt)
    
    X = np.array([kwargs['perc_orang_kaya'], 
                  kwargs['perc_high_rise'],
                  kwargs['perc_commercial'],
                  nearest_ipt_distance[0],
                  kwargs['urban'], 
                  kwargs['suburban'], 
                  kwargs['rural']])
  
    X = scaler.transform(X.reshape(1, -1))
    X = torch.tensor(X, dtype = torch.float32)
    
    predicted_ports = model(X)
    
    y = round(predicted_ports.item())
    
    if y < 0:
      y = 0

    if y % 8 != 0:
      y = 8 * math.ceil(y/8)

    cost_urban_suru = np.array([430.35, 491.15, 491.15])
    urban_suru = np.array([kwargs['urban'],  kwargs['suburban'], kwargs['rural']])
    total_cost = round(y * np.sum(np.matmul(urban_suru, cost_urban_suru)), 2)

    rev_home_biz = np.array([129.0*12, 139.0*12])
    product_home = list()
    product_biz = list()
    
    incremental_factors = kwargs['incremental_factors']

    perc_home = 1 - kwargs['perc_commercial']
    max_home = round(y * perc_home)
    max_biz = round(y * kwargs['perc_commercial'])
    
    total_home = 0.0
    total_biz = 0.0
    product_home.append(total_home)
    product_biz.append(total_biz)

    for i in range(7):
      total_home += round(y * incremental_factors[i] * perc_home)
      total_biz += round(y * incremental_factors[i] * perc_home)

      if total_home >= max_home:
        total_home = max_home

      if total_biz >= max_biz:
        total_biz = max_biz

      product_home.append(total_home)
      product_biz.append(total_biz)

    unit_revenue = np.array([product_home, product_biz])  
    total_revenue = np.dot(unit_revenue.T, rev_home_biz)

    usp = list()
    for revenue in total_revenue:
      usp.append(0.06 * revenue)
    usp = np.array(usp)


    operation_cost = list()
    inflation_rate = 0.04
    for i in range(1, 8):
      operation_cost.append(0.1 * total_cost * math.pow(1+inflation_rate, i))
    operation_cost = np.array(operation_cost)


    #usp = np.insert(usp, 0, 0)
    operation_cost = np.insert(operation_cost, 0, 0)

    opex = usp + operation_cost
    np.put(opex, 0, total_cost)
    #total_revenue = np.insert(total_revenue, 0, 0)
    net_cash_flow = total_revenue - opex
    
    pv_cash_flow = 0
    pv_cash_flows = list()
    cumulative_cash_flow = list()



    for i, cash_flow in enumerate(net_cash_flow):
      discounted_rate = 1/math.pow((1+0.12), i)
      pv = cash_flow * discounted_rate
      pv_cash_flow += pv

      pv_cash_flows.append(pv)
      cumulative_cash_flow.append(pv_cash_flow)

    payback_period = 0
    for i, cash_flow in enumerate(cumulative_cash_flow):
      if cash_flow > 0:
        payback_period = i-1 + (abs(cumulative_cash_flow[i-1]) / pv_cash_flows[i])
        break

    irr = npf.irr(net_cash_flow)

  

    
    
    return y, total_cost,  irr* 100, payback_period
    
    




                  
 
    


