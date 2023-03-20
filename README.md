#  Coupled Attention Networks for Multivariate Time Series Anomaly Detection

This is the implementation of "CAN" based on PyTorch. 

## structure of the code:  

- `lib` folder: some training methods and evaluation methods from [mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch); 
- `models` folder: specific implementation of "CAN" model;
- `utils.py`: method of loading data;  
- `train.py`: train and run the model;  
- `run.sh`: shell script for running models.  
 
You can use `source run.sh $gpu_id $dataset_name` command to run the code, such as `source run.sh 0 SWaT_10`. The dataset can be obtained through [mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch) and [iTrust](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/).
