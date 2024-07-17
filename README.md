# Project: TV-Script Generator

In this project, I develop an algorithm for a generating my own [Seinfield](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNN.

## Getting Started

In this notebook, I will make the first steps towards developing an algorithm that could be used to generate a "fake" script based on the [Seinfield](https://en.wikipedia.org/wiki/Seinfeld) script. At the end of this project, the code will accept a prime word (word for starting the script) and would generate a bunch of script.


### Prerequisites

Thinks you have to install or installed on your working machine:

* Python 3.7
* Numpy (win-64 v1.15.4)
* Pandas (win-64 v0.23.4)
* Matplotlib (win-64 v3.0.2)
* Jupyter Notebook
* Torchvision (win-64 v0.2.1)
* PyTorch (win-64 v0.4.1)

### Environment:
* [Miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

### Installing

Use the package manager [pip](https://pip.pypa.io/en/stable/) or
[miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) to install your packages.  
A step by step guide to install the all necessary components in Anaconda for a Windows-64 System:
```bash
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
pip install torchvision
conda install -c pytorch pytorch
```

## Jupyter Notebook
* `dlnd_tv_script_generation.ipynb`

This jupyter notebook describe the whole project from udacity, from the beginning to the end.

## Download the Datasets

To train and test the model, you need this dataset:

* The [Seinfeld Script](https://github.com/musajoshua/Tv-Script-Generator/blob/master/data/Seinfeld_Scripts.txt).
Place it in this project's home directory, at the location `/data`.


## Running the project

The whole project is located in the file `dlnd_tv_script_generation.ipynb` and it's include the training and the generation part.

### Model architecture

I chose a network architecture that consist of an embedding layer, a multi-layer LSTM and a fully connected output layer.

For data parameters
```python
# Sequence Length
sequence_length =   20 # of words in a sequence
# Batch Size
batch_size = 256
```

For training parameters 
```python
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001
```

For model parameters
```python
# Vocab size
vocab_size = len(vocab_to_int) #vocab_to_int dict
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 250
# Hidden Dimension
hidden_dim = 300
# Number of RNN Layers
n_layers = 3
```

Defining the Model
```python
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

RNN(
  (embedding): Embedding(21388, 250)
  (lstm): LSTM(250, 300, num_layers=3, batch_first=True, dropout=0.5)
  (fc): Linear(in_features=300, out_features=21388, bias=True)
)
```
### Loss function and optimizer

```Python
# loss function and optimizer for normal output
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
```
I use the `CrossEntropyLoss` function and the `Adam` optimizer to train the model.

### Train the model

To train the neural network (RNN), run the file `dlnd_tv_script_generation.ipynb`.


### Output of training

I trained for 10 Epochs

```bash
Training for 10 epoch(s)...
Epoch:    1/10    Loss: 5.3177469187736515

Epoch:    2/10    Loss: 4.263239862725134

Epoch:    3/10    Loss: 3.935509466967499

Epoch:    4/10    Loss: 3.751399925444409

Epoch:    5/10    Loss: 3.614388144635757

Epoch:    6/10    Loss: 3.504066791474186

Epoch:    7/10    Loss: 3.4208360144779184

Epoch:    8/10    Loss: 3.3506077984917484

Epoch:    9/10    Loss: 3.294824342360758

Epoch:   10/10    Loss: 3.2396509992367846
```

After 10 epochs, I got a loss of `3.2396509992367846`. That's god and can be improved.

## Improvements

The next steps will be:
* Fine tune the model more and also increase training time.
* Implement and provide this model as an API using flask.
* Revamp and clean up the code
* Implement with different training parameters (lr, optimizer etc)

## Authors

* Musa Joshua

## License
[MIT](https://choosealicense.com/licenses/mit/)
