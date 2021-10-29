import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.graph_objects as go
import predict
import streamlit as st
from streamlit_folium import folium_static
import folium

with st.sidebar:
  area = st.text_input(f'Enter the name of the area?')
  with st.form("my_form"):
      st.title('Network Capacity Planning')
  
      m = folium.Map(location=[1.937344, 103.366585])
      m.add_child(folium.LatLngPopup())
      folium_static(m)
      st.write('Please copy paste the longitude and latitude values into the columns below.')
      longitude = st.text_input(f'Longitude of the {area}')
      latitude = st.text_input(f'Latitude of the {area}')
      
      perc_orang_kaya = st.slider(f'What is percentage of high income earners in {area}?',
                         min_value = 0.0, max_value = 100.0)
      
      perc_commercial = st.slider(f'What is percentage of commercial in {area}?',
                         min_value = 0.0, max_value = 100.0)
      
      type_community = st.radio(f'What is the type of community in {area}?',
                            ('urban', 'suburban','rural'))
      
      perc_high_rise = st.slider(f'What is percentage of high rises in {area}?',
                         min_value = 0.0, max_value = 100.0)
      
      manpower = st.text_input(f'What is the total manpower cost?')
      material = st.text_input(f'What is the total material cost?')
      incidental = st.text_input(f'What is the total incidental cost?')
  
      submitted = st.form_submit_button('Submit')
    
    
if submitted:
  st.title(f"Automatic Network Planning in {area}")
  
  if type_community == 'urban':
    urban, suburban, rural = 1, 0, 0
  elif type_community == 'suburban':
    urban, suburban, rural = 0, 1, 0
  else:
    urban, suburban, rural = 0, 0, 1
    
       
  total_ports, total_cost, ebit, roi = predict.predict(perc_orang_kaya = float(perc_orang_kaya/100), 
                                                       perc_high_rise = float(perc_high_rise/100), 
                                                       perc_commercial = float(perc_commercial/100),
                                                       latitude = float(latitude),
                                                       longitude = float(longitude),
                                                       urban = urban, 
                                                       suburban = suburban, 
                                                       rural = 1,
                                                       manpower = float(manpower), 
                                                       material = float(material), 
                                                       incidental = float(incidental))
  
  
  
  with st.container():
    commercial_ports = round(perc_commercial/100 * total_ports)
    home_ports = round(total_ports-commercial_ports)
    labels = ['Home','Business']
    values = [ home_ports , commercial_ports]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig)
  
  r1c1,r1c2, r1c3 = st.columns(3)
  r1c1.metric("Home", home_ports)
  r1c2.metric("Total No of Port", total_ports)
  r1c3.metric("Business", commercial_ports )
  
  col1, col2, col3= st.columns(3)
  col1.metric("Cost of Investment", total_cost)
  col2.metric("EBIT", ebit)
  col3.metric("ROI", roi)



 
  
      
                  
    