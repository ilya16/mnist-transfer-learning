# Model Tuning and Transfer Learning on a MNIST dataset

A DNN model for MNIST dataset written using a PyTorch framework.

The task is divided into two parts:
1. Tuning a Deep Learning Model on MNIST digits from 0 to 4
2. Transfer Learning on digits 5 to 9 of a pretrained model from Part 1.

[Jupyter Notebook](solution.ipynb) provides the step by step description of the solution. 

If you would like to train a model or run a transfer learning in a Docker container, 
refer to the following sections on building and running. 

## Fast Run

To run both parts step by step use the following commands:


```sh
git clone https://github.com/ilya16/mnist-transfer-learning
cd mnist-transfer-learning
chmod +x ./run.sh
./run.sh 
``` 

At first, a new model will be trained on digits 0-4. 
Then, the best model state will be used to train a second model on digits 5-9.

## Build

Build a Docker image from the sources with:

```sh
git clone https://github.com/ilya16/mnist-transfer-learning
cd mnist-transfer-learning
docker build -t mnist-transfer-learning .
``` 

or pull image from DockerHub with:
```sh
docker pull ib16/mnist-transfer-learning:latest
```

## Running

### Training a tuned model

Train a new model on digits 0-4 using

```sh
docker run -t --rm -v volume-mnist:/app/state mnist-transfer-learning --fit
```


### Transfer Learning

Run Transfer Learning on digits 5-9 using:

```sh
docker run -t --rm -v volume-mnist:/app/state mnist-transfer-learning --transfer \
    --epochs=100 --early-stopping=25
```


*Note: transfer learning can start only after fitting the model on digits 0-4.*