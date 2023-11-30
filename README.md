# Meta Learning with MAML
## Introduction
This is a MAML implementation for Meta Learning.  

![image](https://github.com/byunghyun23/meta-learning/blob/main/assets/fig1.png)  

## Dataset
We used Omniglot dataset.  
The model is trained using support and query data.  

![image](https://github.com/byunghyun23/meta-learning/blob/main/assets/fig2.png)  
You can see that the data has been downloaded to
```
--dataset
```
by running the following in run.py
```python
# Get DataLoaders
train_dataloader, val_dataloader, test_dataloader = get_dataloader(config)
```

## Run(Train, Test)
```
python run.py
```
After training, you can check the training process by Pyplot.  
And the MAML model file is created
```
--saved_model
```

## Results
![image](https://github.com/byunghyun23/meta-learning/blob/main/assets/fig3.png)
