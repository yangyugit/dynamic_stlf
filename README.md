# A dynamic ensemble method 
This code is possible to create a dynamic ensemble method for most modelling and forecasting problem. Here, we present a residential short-term load forecasting example. 

## getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
You just need MATLAB. This code has been developed on MATLAB R2021b, but should also be compatible with earlier versions. But, to accelerate the training of the deep learning neural network, you should need a GPU. 

### Installing 
The installation in MATLAB is rather simple. Download the file and unzip it, then run the .m code. 

### instruction
#### Dataset:
energydata_complete.csv is a residential load dataset. 
household_power_consumption.txt is also a residential load dataset. 
Both can be found in the UCI machine learning repository. 
The URL of IHEPC is https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption;
The URL of AEP is https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

#### script.m:
data_preprocessing.m deals with the data of IHEPC and AEP to interpolate the missing data and produces the corresponding .mat file which can be load to the MATLAB fast. 

hyperparameters_selection.m illustates the hyperparameters selection of the base models. The main idea is arial and error. Running of the script will take lots of time. 

test_case_1 file contains the corresponding script.m of test scenario 1 that the base model is accurate. 
case_1_aep_error_indices.csv and case_1_ihepc_error_indices.csv show the residential load forecasting performance, that can be obtained by running ensemble_forecasting.m.
The .png figures are the performance figures that can be obtained by running figure_plot.m.

For the base models,
lstm_training.m illustrates the training of LSTM neural network. 
forecast_by_single_method.m includes the VAR, GPR, and LSTM forecasting results. 
bagging_cnn.m presents the bagging CNNs method. 
fusing_lstm.m shows the fusing LSTMs method. 
trim_agg.m shows the trim aggregation method, which depends the shallow_neural_networks.m to obtain the FNN, ELM, and RBF base models. 
Note that, all the training procedures of the base models take lots of time. 

For the dynamic ensemble method,
ensemble_forecast.m illustrates the dynamic ensemble method. 
static_est.m, pf_dyanmic_est.m are the function in ensemble_forecast.m to relize the state estimation of weight coefficients. 

test_case_2 file has the similar sturcture of test_case_1_file. 

#### Recommendation
To only check the performance of dynamic ensemble method for residential load forecasting performance, you can only run the ensemble_forecasting.m to get the error indices .csv file, and run the figure_plot.m to obtain the preformance figures. 
