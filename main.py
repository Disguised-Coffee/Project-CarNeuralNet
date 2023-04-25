from typing import Tuple, List
import statistics
import random
import copy

from neural import *

DATA_FILE = "testing_data/imports-85.data"

CONVERSION_DICTONARY_INPUTS = []

CONVERSION_DICTONARY_OUTPUTS = []

def check_for_letters(string):
    """
    Returns false if there is no numbers in the string.
    """
    # print(string)
    for i in range(len(string)):
        if string[i].isalpha():
            return True
    return False

def reformat_data(data: List[Tuple[List[float], List[float]]]) -> Tuple[List[float], List[float]]:
    """
    Process each line, first looking for strings to put as numbers and then converting every value to a float

    Later on, track columns in terms of names to convert the strings to names

    To be optimized.

    """
    #Step 1 (for both inputs and Outputs)
    # Find value in which strings exist and replace it with a actual number value
    # Record the possible string values in the column in which strings exist
    # Give the data index based on the list with the recorded string values  
  
    # Format inputs into all numerical values.
    for i in range(len(data[0][0])):
        # Check if the piece of data is a string. (\n values are also numbers too)
        if "?" not in data[0][0][i] and ("\n" not in data[0][0][i] or check_for_letters(data[0][0][i])):
            # check each row and find possible values
            possible_values = []
            for row in range(len(data)):
                # Ignore question marks when making the list
                if data[row][0][i] != "?" and data[row][0][i] not in possible_values:
                    # print("appending ",data[row][0][i])
                    # Give string a numerical value (its index in the list).
                    possible_values.append(data[row][0][i])
            
            #Stored values for later c;
            CONVERSION_DICTONARY_INPUTS.append((i,possible_values))
            # Format values in which new row values are index numbers
            for row in range(len(data)):
                data[row][0][i] = possible_values.index(data[row][0][i]) if data[row][0][i] != "?" else possible_values.index(random.choice(possible_values)) # look at placement.
        # # check if the 
        # if data[0][0][i].isnumeric() or data[0][0][i].isdecimal():
          
    
    # Format outputs
    for i in range(len(data[0][1])):
        #Check if the piece of data is a string. (\n values are also numbers too)
        if "?" not in str(data[0][0][i]) and ("\n" not in data[0][1][i] or check_for_letters(data[0][1][i])):
            # check each row and find possible values
            possible_values = []
            for row in range(len(data)):
                if data[row][1][i] not in possible_values:
                    possible_values.append(data[row][1][i])
            
            #Stored values for later c;
            CONVERSION_DICTONARY_OUTPUTS.append((i,possible_values))

            # Format values in which new row values are index numbers
            for row in range(len(data)):
                data[row][1][i] = possible_values.index(data[row][1][i])
    
    new_data = copy.deepcopy(data)

    #Stepus 2 
    # Convert everything to a float
    for row in range(len(data)):
        for outcome in range(len(new_data[row])):
            for col in range(len(new_data[row][outcome])):
                # print(data[row][outcome][col])
                # Handle "?"
                # If there happens to be a questionmark in the data, just use the median.
                if str(new_data[row][outcome][col]) in "?\n":
                    compare_list = []
                    for x in range(len(new_data)):
                        compare_list.append(float(new_data[x][outcome][col] if str(new_data[x][outcome][col]) not in "?\n" else 0))
                    # print(compare_list)
                    new_data[row][outcome][col] = statistics.median(compare_list)
                else:
                    # if str(new_data[row][outcome][col]) in ["two","four"]:
                        # print("ROw",row,"Col",col)
                        # print(new_data[row][outcome][col])
                    new_data[row][outcome][col] = float(new_data[row][outcome][col]) 
    return new_data

def parse_line(line: str, inputs: List[int], outputs: List[int]) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string
        outputs


    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    # print(tokens)
    
    
    # Make inputs based on parameters
    inpt = []
    for i in inputs:
        inpt.append(tokens[i])
    
    outpt = []
    for i in outputs:
        outpt.append(tokens[i])

    return (inpt, outpt)


# Imported from neural_net data
def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    # inputs
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])

    # outputs
    leasts = len(data[0][1]) * [100.0]
    mosts = len(data[0][1]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            if data[i][1][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][1][j] > mosts[j]:
                mosts[j] = data[i][1][j]

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            data[i][1][j] = (data[i][1][j] - leasts[j]) / (mosts[j] - leasts[j])
    
    return data

def run_neural_net(inputs: List, outputs: List, hidden_nodes: int):
    with open(DATA_FILE, "r") as f:
        training_data = reformat_data([parse_line(line,inputs,outputs) for line in f.readlines() if len(line) > 4])

    print(training_data)
    print(CONVERSION_DICTONARY_INPUTS)
    td = normalize(training_data)
    print(td)

    nn = NeuralNet(len(inputs), hidden_nodes, len(outputs))
    nn.train(td) # , iters=100_000, print_interval=1000, learning_rate=0.1)

    #Do Test case stuff here c;

if __name__ == "__main__":
    with open(DATA_FILE, "r") as f:
        training_data = reformat_data([parse_line(line,[0,1,2,3,4,5],[25]) for line in f.readlines() if len(line) > 4])

    # print(training_data)
    print(CONVERSION_DICTONARY_INPUTS)
    td = normalize(training_data)
    print(td)
    # exit()

    nn = NeuralNet(6, 3, 1)
    nn.train(td) # , iters=100_000, print_interval=1000, learning_rate=0.1)

    for i in nn.test_with_expected(td):
        print(f"desired: {i[1]}, actual: {i[2]}")
