# Automatic Sleep Staging with Deep Learning Methods

Automatic Sleep Staging with Deep Learning Methods

## Getting Started

This repo contains the data processing and model to conduct automatic sleep staging. PySpark is used for data preprocessing and EDA. PyTorch is used for hand made RNN and CNN-RNN model.

### Prerequisites

What things you need to install the software

```
Numpy, matplotlib, pytorch, pyspark, pandas
```
## Introductions

data_processor.py contains the code for data preprocessing 

utils.py contains all the functions that might be used during the process, including data process function and confusion matrix function (for modeling part)

data_accessor.py is the object to load data for models and EDA

sleep_dataloader.py is the data loader object for PyTorch models

sleep_dataset.py is the data set object for PyTorch models

data_set.py defines the enums used to specify train, test and val data for dataloader and dataset

cnn_rnn_data_post_processor.py and rnn_only_data_post_processor.py is the padding post processor for two models, CNN-RNN model and hand RNN model respectively

sleep_rnn.py, sleep_rnn_cnn.py, sleep_cnn_only.py are the three model objects for PyTorch, where sleep_rnn is the Seq2Seq model, which is shared between CNN-RNN model and Hand RNN model. sleep_cnn_rnn is for the CNN-RNN model. And sleep_cnn_only is for pretraining CNN model.

padded_loss.py is the the loss function used in the process. Given the variable length nature of RNN model, we have to take care of it when calculate cross entropy loss.

base_trainer.py is the base class for the training process. It contains many important shared functions during the modelling process, including training, validation, testing, save loss, publishing to tensorboard etc. hand_rnn_trainer.py is its derived class, used for training and testing for Hand RNN model. cnn_trainer.py is also one of its derived class, used for pretraning of CNN part. cnn_rnn_trainer.py is also the derived class of it and is used for training and testing of CNN-RNN model.

The main three entry points for the models are, run_cnn_only.py is to start the process for pretraining CNN. run_hand_rnn.py is to start the process for Hand RNN model. run_cnn_rnn.py is to start the process for CNN RNN model.


## Running the process

First thing to make sure the process run smoothly is to ensure you have created log, parameter, loss, plot and data folders in the directory. Also the raw data should be sitting in sleep-edf-database-expanded-1.0.0 folder.

To run data preprosessing, you should go to data_processor.py.
```
processor = DataProcessor()
processor.save_file_spark()
```
this is to parse the raw file and save transformed data, please note that some of the parse of raw data will fail, you need to add them in removed_subject, you can find more details in code comments.

```
processor = DataProcessor()
processor.load_file_spark()
```
this is to separate transformed data to bucket of epochs, so it's ready for hand RNN model.

```
processor = DataProcessor()
processor.load_eeg_file_spark()
```
this is to separate transformed data to bucket of epochs, so it's ready for RNN RNN model.


Entry point to run the models, as mentioned earlier, is the three run script, run_cnn_only.py, run_hand_rnn.py, run_cnn_rnn.py. And they all share the similar structure, we will only take one example to show how to run them here.

Before you run the process, you can change the hyperparameter during the initialization of the trainier object. You can use the config class to define the value of hyperparameter if you want.

To train the model, simply type train as input parameter.
```
run_hand_rnn.py train
```

To get the confusion matrix, simply type test as input parameter but please remember to include --steps so the trainer know which specific parameter to be loaded. During training, the model will dump the parameter for every given steps, which can be loaded here to test performance.
```
run_hand_rnn.py test --steps=175
```

Also please note, CNN RNN model will be slightly different in that before it initializes its own trainer, we need to initialize a cnn trainer too and we have to specifiy number of steps in CNNRNNTrainer so that it knows which specific CNN model to be loaded before the joint model training starts.

## Authors

* **Sizhang Zhao** - *Initial work* 


