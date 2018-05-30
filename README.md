# Neural-Network-Development
Neural Network project for forecasting Historical Stock market data sets. This project will contain two networks, firstl the Recurrent Neural Network and secondly a Long Short-Term Memeory Network which both will use processed stock data to predict closing prices from normal sets.

## Project Dependancies
**Python (3.6) 64-bit Version;**
https://www.python.org/download/releases/3.4.0/

**Numpy(1.14);**
https://docs.scipy.org/doc/numpy-1.14.0/reference/,

**TensorFlow(1.4);**
https://www.tensorflow.org/

**Matplot Library(0.1.0);**
https://matplotlib.org/

**Visual Studio Enterprise 2017;**
https://www.visualstudio.com/vs/enterprise/
(Pending development to independant executable)

## Python Dependancies
**Math,**

**CSV,**

**Operator,**

**Random**

"Pending reduction of current dependancies, The above imports are from Python 3.0 library and do not require updating from the files."

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

## __Training__
Training will deploy a Droppout Wrapper supporting funcitonality to decrease training time and approach more accurate values quicker than standard operation. This is Accomplished by incrementing a propability counter which reflects the propability of neuron cells within the Network of being dropped out of the calculation, this within the project for training is automatically predetermined as **1.0** which indicates an absoulte drop of values within each Neural Network by default. This is to decrease the training time frame and reduce ware on the host machines processing power, which is severally taxing.

The Finial State of the network is automatically presumed to be the most accurate, this means when the **Epoch**(Total number of complete dataset cycles inside the network) Cycle is completed the finial state of the network is abstracted and saved through the TensorFlow Save feature, which stores the data to a .CKPT File.

## __Testing__
Testing is an identical structure to the Training(As this allows the state saved .CKPT data to be reintroduced into the Network), the Droppout Wrapper as previously mentioned however is dropped from this network. The Network will preform only a **SINGLE EPOCH** Cycle and store the data output from the cycle as the testing data.

# Program Operation
## Program Operation Structure
When variables input is set in the **Run.py** Class, it is then deployed to create the Network.
(Warning: There is little user validation here and WILL cause issues, do not manipulate variables unless knowledge is present to their operation within the project)

**Intialize**

**RNN_Model.py // Class RNN() // def __init__();**

> (1) Commint User Declared Variables to Memory

> (2) Acquire user selected activation function(TensorFlow Library Code Grab)

> (3) Acquire Seperated data from user targeted file from Dataset Folder(Seperates; Plotter, Input and Label Data)

> (4) Data Toggle Switch:
>- Toggle One Selected; Standard Operation, Prediction is [Open][Low][High] To predict [Close] of Same trading day
>- Toggle Two Selected; Simplified Operation, Prediction is [Close] To Predict [Close] of the Following tading day
>- Preprocesses Acquired Data (3) To toggle selection

> (5) Data Normalization, (Value-Max)/(Min-Max) to reduce values between zero and One.

> (6) Filereading, Stores ALL PRIOR DATA to File solutions (CSV)

> (7) Automatic Sizing of Training/Testing data for relay to TensorFlow Tensors(Variables, Specifically 'Placeholders')

> (8) Automatic Sizing of Batches for Batch Processing 

> (9) **Plotter Call**, Plots the graph of the Raw Input File

> (9) Session Variable Declare for Neural Networks

> (10) Detail feedback for user(Reflects; Input Count, Output Count, Total Dataset Size, Training Data Size, Batching Details, Testing Data Size)

## Automated Functions
There are a vast amount of functions which are automated within the project, these features are predetermined and fixed as part of the project. The Following features are automated;
- Batching: Batch Size is automatically determined for best fit(How many batches and how much data in each batch)
- FileReading: Reads target datafile, presumed to conform to the download datasets from target website(No Validation, No Null Checker)
- Tensors: Tensorflow Variables are automatically scaled based upon user input and dataset injection.
- Dataset Sizing: Declared split by the User is applied automatically of Training and Testing
- Session Arrays: All data is handeled automatically between Training and Testing and requires no user intervention
- Training: Fully Automated from user input, no direct interation required
- Testing: Fully Automated from user input, no direcit interation required
- Plotting: Fully Automated from user input, no direct interation required 
- Tensor Saving: Reloading and Saving fully automated, no direct interation required

## Known Project Issues
There are some known issues which present operation problems, this list will provide the recapped version to aid external operation;
- Resource Intensive, the Original machine operates a (i7-7700k CPU & 32GB of DDR4 @2400mhz) and experianced extreme strain while running networks, Larger Networks (Layers 5-10 & Neurons 50-100) Will perform extremly slowly while computating. This project has not considered smoother operation primarlly due to the skill level of the programmer and that the optimizations would be targeted for the current machine configuration rather than general operation.
- Null Field Values, the user targeted file, presumed to be a .CSV file performs no preprocessing prior to information grabbing, this means if there is a NULL value or any other value not permited the project will through an ERROR and not Run. This is a very confusing issue which presents no work around, methods deployed to perform this did not operate correctly or even at all and had to be dropped to maintain time constraints of project delevaries. These will have to be removed by hand untill a functional work around it constructed.
- Batch Processing, the feature has been successfully implemented and operates correctly, however batch processing causes dramatic feedback issues upon the Neural Network. The Networks in this project are regression based taskings, which requires connetivity of the entire dataset (Left to Right, Complete) rather than batching in order to perform predictions. No work around is known to the programmer (Limited Experiance) at this time to adjust this, the values of batch process are fixed to One as of this update.
- Plotter Issues, due to inexperiance of the programmer possess of the Matplot Library toolset the project runs into issues when plotting the Bi-Directional Graph models 


