#Server code for running machine learning code
import dill

import pandas as pd 

from fastapi import FastAPI
from pydantic import BaseModel 


app = FastAPI()
pipe = dill.load(open('./models/customer_satisfaction.pkl', mode = 'rb'))

class Form(BaseModel):
    id : int
    Gender : str
    customer_type : str
    Age : int
    type_of_travel : str
    travel_class : str
    flight_distance : int
    inflight_wifi_service : int
    departure_arrival_time_convenient : int
    ease_of_online_booking : int
    gate_location : int
    food_and_drink : int
    online_boarding : int
    seat_comfort : int
    inflight_entertainment : int
    onboard_service : int
    leg_room_service : int
    baggage_handling : int
    checkin_service : int
    inflight_service : int
    cleanliness : int
    departure_delay_in_minutes : int
    arrival_delay_in_minutes : float

class Prediction(BaseModel):
    Result : int 

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return pipe['metadata']

@app.post('/predict', response_model = Prediction)
def predict(form : Form):
    df = pd.DataFrame.from_dict([form.dict()])
    df.rename(columns = {'customer_type' : 'Customer Type', 'type_of_travel' : 'Type of Travel', 'travel_class' : 'Class', 'flight_distance' : 'Flight Distance', 'inflight_wifi_service' : 'Inflight wifi service', 'departure_arrival_time_convenient' : 'Departure/Arrival time convenient', 'ease_of_online_booking' : 'Ease of Online booking',
                       'gate_location' : 'Gate location', 'food_and_drink' : 'Food and drink', 'online_boarding' : 'Online boarding', 'seat_comfort' : 'Seat comfort', 'inflight_entertainment' : 'Inflight entertainment',
                       'onboard_service' : 'On-board service', 'leg_room_service' : 'Leg room service', 'baggage_handling' : 'Baggage handling', 'checkin_service' : 'Checkin service', 'inflight_service' : 'Inflight service', 
                       'cleanliness' : 'Cleanliness', 'departure_delay_in_minutes' : 'Departure Delay in Minutes', 'arrival_delay_in_minutes' : 'Arrival Delay in Minutes'}, inplace=True)
    y = pipe['model'].predict(df)
    return {
        'Result' : y[0]
    }