% preprocessing the data i.e., dealing with the NaN data and interpolation

%% deal with the IHEPC Dataset
clear all
% load data (the date has on temperature index)
data = readtable('household_power_consumption.txt');
active_power = data.Global_active_power;
time = data.Time;
date = data.Date;
date = string(date,"dd-MMM-yyyy");
time = datetime(date) + time;

% change kilowatt to watt; the same unit to AEP
active_power = active_power.*1000; 

% preproces: linear interp1 with NaN
active_power(isnan(active_power))= interp1(find(~isnan(active_power)), ...
    active_power(~isnan(active_power)), find(isnan(active_power)), 'linear');

save 'ihepc_dataset'

%% deal with the AEP Dataset
clear all
% load data 
data = readtable('energydata_complete.csv');
appliances = data.Appliances;
time = data.date;
dewpoint = data.Tdewpoint;

dewpoint1 = [dewpoint(1); dewpoint(1:end-1)]; % the switch of weather forecasting
series_data = [appliances, dewpoint]; 
series_data = series_data(2:end, :);  time = time(2:end);

save 'aep_dataset'