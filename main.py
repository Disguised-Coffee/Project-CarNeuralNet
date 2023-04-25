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
    Return data with floats.

    Process each line, first looking for strings to put as numbers and then converting every value to a float. Tracks all values in constants CONVERSION_DICTONARY_INPUTS and CONVERSION_DICTONARY_OUTPUTS.

    Fairly useful c;
    """
    # Check 1 Synopsis
    # Find value in which strings exist and replace it with a actual number value
    # Record the possible string values in the column in which strings exist
    # Give the data index based on the list with the recorded string values  

    # Check 2 Synopsis
    # Record the least and greatest values in the column in which strings exist 
    # Give the data index based on the list with the recorded string values  
  
    # Format all values into numerical values (at the end it will be a decimal number).
    for outcome in range(len(data[0])): # 2
        for col in range(len(data[0][outcome])): # for each column
            # Check 1
            # Check if the piece of data is a string. (\n values are also numbers too)
            if "?" not in data[0][outcome][col] and ("\n" not in data[0][outcome][col] or check_for_letters(data[0][outcome][col])):
                # check each row and find possible values
                possible_values = []
                for this_row in range(len(data)):
                    # Ignore question marks when making the list
                    if data[this_row][outcome][col] != "?" and data[this_row][outcome][col] not in possible_values:
                        # print("appending ",data[this_row][0][i])
                        # Give string a numerical value (its index in the list).
                        possible_values.append(data[this_row][outcome][col])
                
                #Stored values for later c;
                if outcome == 0:
                    CONVERSION_DICTONARY_INPUTS.append((col,possible_values))
                else:
                    CONVERSION_DICTONARY_OUTPUTS.append((col,possible_values))
                # Format values in which new row values are index numbers
                for this_row in range(len(data)):
                    data[this_row][outcome][col] = float(possible_values.index(data[this_row][outcome][col]) if data[this_row][outcome][col] != "?" else possible_values.index(random.choice(possible_values))) # look at placement.
            
            # Check 2 format it into a decimal value.
            else:
                # go through the rows in that column, tracking the range of values.

                # HEHEHE
                # basically this is an if-else system for a variable.
                least = data[0][outcome][col] if type(data[0][outcome][col]) == float or (data[0][outcome][col].isnumeric() or data[0][outcome][col].isdecimal()) else data[3][outcome][col]

                greatest = data[0][outcome][col] if type(data[0][outcome][col]) == float or (data[0][outcome][col].isnumeric() or data[0][outcome][col].isdecimal()) else data[3][0][col]
                
                for row in range(len(data)):
                    # If there happens to be a question mark in the non-string data, just replace it with the median!
                    # (Yes this is inaccurate but oh well.)
                    # Yet this is useful as training data...
                    if str(data[row][outcome][col]) in "?\n":
                        compare_list = []
                        # Get possible values.
                        # Btw, if else branches and for loops don't work well in one line.
                        # (pain)
                        for x in range(len(data)):
                            compare_list.append(float(data[x][outcome][col] if str(data[x][outcome][col]) not in "?\n" else 0))

                        data[row][outcome][col] = float(statistics.median(compare_list))
                    else:
                        #finally
                        data[row][outcome][col] = float(data[row][outcome][col]) 
                        
                        print(type(data[row][outcome][col]))
                        # For tracking purposes for numbers
                        if data[row][outcome][col] < float(least):
                            least = data[row][outcome][col]
                        elif data[row][outcome][col] > float(greatest):
                            greatest = data[row][outcome][col]
                
                # Conversion purposes c;
                if outcome == 0:
                    CONVERSION_DICTONARY_INPUTS.append((col,["N", least, greatest]))
                else:
                    CONVERSION_DICTONARY_OUTPUTS.append((col,["N", least, greatest]))
    return data

def parse_line(line: str, inputs: List[int], outputs: List[int]) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string
        outputs


    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    
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
    # for each separated piece of data
    for outcome in range(len(data[0])): # 2
        leasts = len(data[0][outcome]) * [100.0]
        mosts = len(data[0][outcome]) * [0.0]

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                if data[i][outcome][j] < leasts[j]:
                    leasts[j] = data[i][outcome][j]
                if data[i][outcome][j] > mosts[j]:
                    mosts[j] = data[i][outcome][j]

        for i in range(len(data)):
            for j in range(len(data[i][outcome])):
                data[i][outcome][j] = (data[i][outcome][j] - leasts[j]) / (mosts[j] - leasts[j])
    
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

    print(CONVERSION_DICTONARY_OUTPUTS)
    # td = normalize(training_data)
    # print(td)
    exit()

    nn = NeuralNet(6, 3, 1)
    nn.train(td) # , iters=100_000, print_interval=1000, learning_rate=0.1)

    for i in nn.test_with_expected(td):
        print(f"desired: {i[1]}, actual: {i[2]}")
