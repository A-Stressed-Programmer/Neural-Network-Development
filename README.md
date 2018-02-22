#Warning
I am a: Select_One(poor,idiot,sarcastic, asshole) student developing a project, I am also real-stingy to buy private. Please kindly do not judge how God awful some sections of code are typed because I am: Select_One(idiot,special,desperate,ignorant). Also I failed English in my higher, so spelling is gonna be something from a Steven King Horror novel in here.

You have been warned!

# Neural-Network-Development
Neural Network project for forecasting Historical Stock market data sets. This project will contain two networks, firstl the Recurrent Neural Network and secondly a Long Short-Term Memeory Network which both will use processed stock data to predict closing prices from normal sets.

## Project Dependancies
**Numpy(1.14),**
**TensorFlow(1.4),**
**Math,**
**CSV**

## Datasets
The datasets are extracted from Yahoo Finance(Here:https://finance.yahoo.com/quote/AAPL/history?period1=345427200&period2=1519257600&interval=1d&filter=history&frequency=1d), which can be downloaded into .CSV file format by access ing the 'Historical Data' Tab and utilizing the 'Download Data' button. This will download a Stock.CSV file, the main objective of the Project is to design a data_parser in a manor which can allow any dataset in the following format to be introduced into the Neural Network;

**[Date][Open_Stock][High_Stock][Low_Stock][Adj Close][Volume]**

This Build will seperate this data into the following sections;

###### **Input_Data**
[Open_Stock],[High_Stock],[Low_Stock]

###### **Label_Data/Results_Data**
[Close_Stock]

###### **Plotter_x_Axis_data**
[Date]

###### **REMOVE**
[Adj Close**],[Volume]

'Input_Data' represents three datasets from a single trading day from the dataset, these will be injected into the Neural Network to potentionally predict the 'Label_Data' of that same day. This will allow accuracy assessment further into the project. The model will utilize the data as follows;

###### Network Example
(Input_Data[Open_Stock(172.83)][Highest_Stock(174.12)][Lowest_Stock(171.07)]) ---> Neural Network ---> (Output_Data[Prediction(171.10)])

The Project will utilize all variables inside the Stock.CSV File, this creates a three part entry as the 'Input' which will be processed by a network of user defined size to produce a single 'Output' which will be stored as result prediction. As any standard Neural Network there are two sets of processes; __'Training'__ and __'Testing'__ both of which require slightly different approaches.

TBC
