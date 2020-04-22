# LT2212 V20 Assignment 3

# part 1

You can run the `a3_features.py` by providing some commandline arguments. Here is an example:

```
python3 a3_features.py enron_sample output.csv 100
PLEASE NOTE. YOU NEED TO RUN THE SCRIPT IN THE SAME DIRECTORY the enron_samples FOLDER IS. 
```

In this example enron_sample is the folder where the sample texts are. Output.csv is the target to where to output the results and 100 is the number of dimensions.

In the result csv you will see features of each file, the author, and if it was split into the test or training data. The code mainly repeats the assignment 2.

# part 2

Part 2 needs the target csv from part1. You run it simply like this:

```
python3 a3_model.py output.csv 
```

This will train the model without any hidden layers and print accuracy/loss per epoch and reports.

The samples (of two documents to compare) chosen are selected randomly but balanced. 50% of samples of two texts with matching authors and 50% of not matching. Currently the training and test data are split 80/20 with 3200 samples in the training data and 800 samples in the test data.

# part 3

If you run the same python program `a3_model.py` but with some optional arguments it can run with hidden layers.

Example:

```
python3 a3_model.py output.csv --hl 64 --choice 1
```

The `--hl` argument is the number of hidden layers it should run with and the
`--choice` argument takes only 1 or 2. Using 1 will make the program use ReLU activation function and 2 will make it use Softmax. Same training information and reports will be printed as in part 2.



# bonus

You can run the bonus part simply by calling `python3 partbonus.py output.csv plot.png`. This will use the same `output.csv` and the same functions from part3 with the ReLU activation function but will run it several times with different sizes of hidden layers. Then it will generate a `plot.png` file with the plot. Example `plot.png` output from one run is included in the git repository.
The bonus script imports a function from a3_model.py. Please, run in in the same directory. 

# as a note

I managed to reach at maximum around 0.70 accuracy by changing hl size and number for epochs. 

Also, I managed to implement the assingment only using torch.round for output of my model.
The thing is that if y_test array is always 1 or 0, and after prediction list would give floats like [0.74, 0.83, 0.37...] 
the classification report would give an error, as it cannot "handle a mix of binary and continuous targets". 

Maybe I am missing something vital here, but the current implementation is the only way I could solve the assignment.

# results disscusion

The model currently runs with 30 epochs as I see better results with this number. I also see a bit better result for when run with hidden layers, 
however not drastically better.
I also see a bit better result for choice 1 (ReLU).  However all the results don't differ much, I think because of the poor quality of data.

