import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import pandas as pd


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
      
    product_kaya = round(0.4 * y * kwargs['perc_orang_kaya'])
    product_sederhana = round(0.6 * y * kwargs['perc_orang_kaya'])
    product_commercial = round(0.8 * y * kwargs['perc_commercial'])
    balance = round(1.0 * (y - product_kaya - product_sederhana - product_commercial))
    
    revenue = (139.0 * product_kaya + 129.0 * product_sederhana + 139.0 * product_commercial + 89.0 * balance) * 36
    total_cost  = kwargs['manpower'] + kwargs['material'] + kwargs['incidental']
    ebit = round(revenue - total_cost, 2)
    roi = round((ebit/total_cost) * 100, 2)
    
    
    return y, total_cost, ebit, roi
    
    




                  
 
    


