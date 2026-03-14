This project used for determining the price of houses,
basically its a ML(RandomForestRegressor) model,  
which trains on a dataset called california housing dataset and 
for user interactions i used streamlit technology
Additionally for running or setup this profect you need to follow following steps:-
1. Run the main.py which generates model.pkl and pipeline.pkl, run again main.py this time model's performance is checked on input.csv and model creates a csv file      called output.csv in this csv model's output is present in a last column for each record respectively
2. After that, run app.py to get the user interface for interaction, where you can enter inputs to obtain predictions based on the trained model.
