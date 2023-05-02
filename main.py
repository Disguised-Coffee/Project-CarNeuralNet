from typing import Tuple, List
import statistics
import random

from neural import *

DATA_FILE = "testing_data/imports-85.data"
"""Where the main training data exists"""

TEST_CASE_FILE = "testing_data/imports-small.data"
"""Testing file with example test cases"""

CONVERSION_INPUTS : List = []
"""Where inputs exist to be converted"""

CONVERSION_OUTPUTS : List = []
"""Where outputs exist to be converted"""

WRITEN_TO_CONVERSION = False
"""Prevent accidental overwritting of conversion values."""

NEURAL_NETS_RAN = 0
"""Neural net already ran"""

# Important string value to differentiate between string and number pairs.
NUMERICAL_VALUE_STRING = "2656f9b579586eb2b7b2b341ee262d25"
"""Very important"""

UNKNOWN_VALUE = "?"

def reformat_data(data: List[Tuple[List[float], List[float]]]) -> Tuple[List[float], List[float]]:
    """Main course of the entire program.

        Process each line, first looking for strings to put as numbers and then converting 
        every value to a float. Tracks all values in constants CONVERSION_INPUTS and CONVERSION_OUTPUTS.

        Fairly useful c;

    Args:
        line - one line of the CSV as a string
        outputs


    Returns:
        tuple of input list and output list
    """
    # Check 1 Synopsis
    # Find value in which strings exist and replace it with a actual number value
    # Record the possible string values in the column in which strings exist; skip ?
    # If there is a ?, give it a random string value
    # Give the data index based on the list with the recorded string values 
    # Turn that index into a float!

    # Check 2 Synopsis []
    # Record the least and greatest values in the column in which strings exist 
    # Give ? a median value (or a educated random number)
    # Convert the thing to a float!

    # Helper method (Yes, this is an annoying thing. Look up nested functions)
    def hyphen_check(string) -> bool:
        """
        Returns true if it is part of a word (letter next to hyphen).
        """
        # print(string)
        hypen_location = string.find("-")
        return string[hypen_location+1: hypen_location + 2].isalpha()
    
    global WRITEN_TO_CONVERSION
    # Format all values into numerical values (at the end it will be a decimal number).
    for outcome in range(len(data[0])): # 2
        for col in range(len(data[0][outcome])): # for each column
            # Check 1
            # Check if the piece of data is a string. (\n values are also numbers too)
            # 1) no question mark 2) is not a number.
            if UNKNOWN_VALUE not in data[0][outcome][col] and (data[0][outcome][col].isalpha() or hyphen_check(data[0][outcome][col])):
                # check each row and find possible values
                if not WRITEN_TO_CONVERSION:
                    possible_values = []
                    for this_row in range(len(data)):
                        # Ignore question marks when making the list
                        if data[this_row][outcome][col] != UNKNOWN_VALUE and data[this_row][outcome][col] not in possible_values:
                            # print("appending ",data[this_row][0][i])
                            # Give string a numerical value (its index in the list).
                            possible_values.append(data[this_row][outcome][col])
                    
                    #Stored values for later c;
                    if outcome == 0:
                        CONVERSION_INPUTS.append((col,possible_values))
                    else:
                        CONVERSION_OUTPUTS.append((col,possible_values))
                # Format values in which new row values are index numbers
                for this_row in range(len(data)):
                    # Use the Coversion list instead of the possible values.
                    # data[this_row][outcome][col] = float(possible_values.index(data[this_row][outcome][col]) if data[this_row][outcome][col] != "?" else possible_values.index(random.choice(possible_values))) # look at placement.
                    if outcome == 0:
                        data[this_row][outcome][col] = float(CONVERSION_INPUTS[col][1].index(data[this_row][outcome][col]) if data[this_row][outcome][col] != "?" else round(random.random() * len(CONVERSION_INPUTS[col][1]))) # look at placement.
                    else:
                        data[this_row][outcome][col] = float(CONVERSION_OUTPUTS[col][1].index(data[this_row][outcome][col]) if data[this_row][outcome][col] != "?" else round(random.random() * len(CONVERSION_OUTPUTS[col][1]))) # look at placement.
            
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
                    if str(data[row][outcome][col]) in (UNKNOWN_VALUE + "\n"):
                        compare_list = []
                        # Get possible values.
                        # Btw, if else branches and for loops don't work well in one line.
                        # (pain)
                        for x in range(len(data)):
                            compare_list.append(float(data[x][outcome][col] if str(data[x][outcome][col]) not in (UNKNOWN_VALUE + "\n") else 0))

                        data[row][outcome][col] = float(statistics.median(compare_list))
                    else:
                        #finally
                        data[row][outcome][col] = float(data[row][outcome][col]) 
                        
                        # print(type(data[row][outcome][col]))
                        # For tracking purposes for numbers
                        if data[row][outcome][col] < float(least):
                            least = float(data[row][outcome][col])
                        elif data[row][outcome][col] > float(greatest):
                            greatest = float(data[row][outcome][col])
                
                # Conversion purposes c;
                if not WRITEN_TO_CONVERSION:
                    if outcome == 0:
                        CONVERSION_INPUTS.append((col,[NUMERICAL_VALUE_STRING, least, greatest]))
                    else:
                        CONVERSION_OUTPUTS.append((col,[NUMERICAL_VALUE_STRING, least, greatest]))
    WRITEN_TO_CONVERSION = True
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
def normalize(data: List[Tuple[List[float], List[float]]]) -> List[Tuple[List[float], List[float]]]:
    """Legacy version of the normalize function. 
        (Makes the data range for each input feature from 0 to 1)

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)

    """
    for outcome in range(len(data[0])): # 2
        leasts = len(data[0][outcome]) * [100.0]
        mosts = len(data[0][outcome]) * [0.0]

        # For each row
        for i in range(len(data)):
            # for each column
            for j in range(len(data[i][outcome])):
                if data[i][outcome][j] < leasts[j]:
                    leasts[j] = data[i][outcome][j]
                if data[i][outcome][j] > mosts[j]:
                    mosts[j] = data[i][outcome][j]

        for i in range(len(data)):
            for j in range(len(data[i][outcome])):
                data[i][outcome][j] = (data[i][outcome][j] - leasts[j]) / (mosts[j] - leasts[j])
                # print(leasts[j])
    
    return data
    
def normalize_exp(data: List[Tuple[List[float], List[float]]]) -> List[Tuple[List[float], List[float]]]:
    """Makes the data range for each input feature from 0 to 1 based on conversion inputs.

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)

    """
    # for each separated piece of data
    for outcome in range(len(data[0])): # 2  
        #Depending on the side of the data we are converting, we use two different lists
        if outcome == 0:
            for col in range(len(data[0][outcome])):
                if CONVERSION_INPUTS[col][1][0] == NUMERICAL_VALUE_STRING:
                    for row in range(len(data)):
                        # data[i][outcome][j] = (data[i][outcome][j] - leasts[j]) / (mosts[j] - leasts[j])
                        # print(type(data[row][outcome][col]))
                        # print(CONVERSION_INPUTS[col][1][2])
                        data[row][outcome][col] = (data[row][outcome][col] - float(CONVERSION_INPUTS[col][1][1])) / (float(CONVERSION_INPUTS[col][1][2]) - float(CONVERSION_INPUTS[col][1][1]))
                else:
                    length = len(CONVERSION_INPUTS[col][1])
                    for row in range(len(data)):
                        data[row][outcome][col] = data[row][outcome][col] / float(length)
                    
        if outcome == 1:
            for col in range(len(data[0][outcome])):
                
                if CONVERSION_OUTPUTS[col][1][0] == NUMERICAL_VALUE_STRING:
                    for row in range(len(data)):
                        # data[i][outcome][j] = (data[i][outcome][j] - leasts[j]) / (mosts[j] - leasts[j])
                        data[row][outcome][col] = (data[row][outcome][col] - float(CONVERSION_OUTPUTS[col][1][1])) / (float(CONVERSION_OUTPUTS[col][1][2]) - float(CONVERSION_OUTPUTS[col][1][1]))
                else:
                    length = len(CONVERSION_OUTPUTS[col][1])
                    for row in range(len(data)):
                        data[row][outcome][col] = data[row][outcome][col] / float(length)
    return data

#Denormalize C;
def denormalize(data: List[Tuple[List[float], List[float]]]) -> List[Tuple[List[float], List[float]]]:
    """Converts and entire dataset of values of 0 through 1 into actual values based on the conversion lists.
    
        ~ Thought normalizing was bad? It actually isn't. ~
    
        ~ In normalize, we just subtract the least value and then divide it by the range~
    
        ~ For this, we just do the reverse! ~

    Args:
        data - list of (input, output) tuples

    Returns:
        denormalized data where input features are in strings and floats

    """
    for outcome in range(len(data[0])): # 2  
        #Depending on the side of the data we are converting, we use two different lists
        if outcome == 0:
            for col in range(len(data[0][outcome])):
                # Floats
                if CONVERSION_INPUTS[col][1][0] == NUMERICAL_VALUE_STRING:
                    least = float(CONVERSION_INPUTS[col][1][1])
                    most = float(CONVERSION_INPUTS[col][1][2])
                    # Convert each row.
                    for row in range(len(data)):
                        data[row][outcome][col] = (data[row][outcome][col]  * (most - least)) + least
                # Strings
                else:
                    length = len(CONVERSION_INPUTS[col][1])
                    data[row][outcome][col] = CONVERSION_INPUTS[col][1][int((data[row][outcome][col]  * (most - least)) + least)]
                    
        if outcome == 1:
            for col in range(len(data[0][outcome])):
                # Floats
                if CONVERSION_OUTPUTS[col][1][0] == NUMERICAL_VALUE_STRING:
                    least = float(CONVERSION_OUTPUTS[col][1][1])
                    print(least)
                    most = float(CONVERSION_OUTPUTS[col][1][2])
                    print(most)
                    for row in range(len(data)):
                        # data[i][outcome][j] = (data[i][outcome][j] - leasts[j]) / (mosts[j] - leasts[j])
                        # print(type(data[row][outcome][col]))
                        # print(CONVERSION_INPUTS[col][1][2])
                        data[row][outcome][col] = (data[row][outcome][col]  * (most - least)) + least
                # Strings
                else:
                    length = len(CONVERSION_OUTPUTS[col][1])
                    data[row][outcome][col] = CONVERSION_OUTPUTS[col][1][int(round(data[row][outcome][col] * float(length)))]
    return data

def denormalize_output(data: List[float]) -> List[float]:
    """Converts a list (preferably outputs) of values of 0 through 1 into actual values based on the conversion lists.

        ~ mini function <; ~
    Args:
        data - list of outputs.

    Returns:
        denormalized data where input features are in strings and floats

    """
    for col in range(len(data)):
        if CONVERSION_OUTPUTS[col][1][0] == NUMERICAL_VALUE_STRING:
            least = float(CONVERSION_OUTPUTS[col][1][1])
            most = float(CONVERSION_OUTPUTS[col][1][2])

            data[col] = (data[col]  * (most - least)) + least
        else:
            length = len(CONVERSION_OUTPUTS[col][1])
            data[col] = CONVERSION_OUTPUTS[col][1][int(round(data[col] * float(length)))]
    return data

def reset_conversions():
    global CONVERSION_INPUTS
    CONVERSION_INPUTS.clear()
    global CONVERSION_OUTPUTS
    CONVERSION_OUTPUTS.clear()
    global WRITEN_TO_CONVERSION
    WRITEN_TO_CONVERSION = False

def run_neural_net(inputs: List, outputs: List, hidden_nodes: int, test_cases: List = [TEST_CASE_FILE], rounding_factor: int = 3, iters :int = 1000, print_inter = 100, learning_rate = 0.5) -> None:
    """
    Runs the neural net program.

    ~!~ Just puts some input values and desire outputs, and get something! ~!~
    """
    # Helper function to round test_case values from neural net
    def round_each(values: List):
        if rounding_factor == None:
            return values
        else:
            for i in range(len(values)):
                if type(values[i]) != str:
                    values[i] = round(values[i], rounding_factor)
            return values
    
    # RESET EVERYTHING.
    reset_conversions()

    # import global constants
    global NEURAL_NETS_RAN
    global CONVERSION_INPUTS
    global CONVERSION_OUTPUTS

    print("*" * 65,"\n\t\tStarting neural net number", NEURAL_NETS_RAN,"\n" + "*" * 65)
    with open(DATA_FILE, "r") as f:
        training_data = reformat_data([parse_line(line,inputs,outputs) for line in f.readlines() if len(line) > 4])
    
    # print(training_data)
    print("\nINPUTS")

    for x in CONVERSION_INPUTS:
        print(x)
    print("\nOUTPUTS")
    
    for x in CONVERSION_OUTPUTS:
        print(x)
    print("\n")

    td = normalize_exp(training_data)
    # print(td)
    
    print("~ running neural net ~ \n")
    nn = NeuralNet(len(inputs), hidden_nodes, len(outputs))
    nn.train(td, iters=iters, print_interval=print_inter, learning_rate=learning_rate)
    
    for test_case_file_path in test_cases:
        print("\n ~ Evaluating desired test case at "+ test_case_file_path +" ~\n")
        #Do Test case stuff here c;
        with open(test_case_file_path, "r") as f:
            testing_data = reformat_data([parse_line(line,inputs,outputs) for line in f.readlines() if len(line) > 4])
        
        for i in nn.test_with_expected(normalize_exp(testing_data)):
            print(f"Desired: {denormalize_output(i[1])}, Actual: {round_each(denormalize_output(i[2]))}")

    print("\n \t\t~ end of neural net",NEURAL_NETS_RAN," ~ \n")
    NEURAL_NETS_RAN += 1
    
if __name__ == "__main__":
    """
    to do:
    Make neural net for...
    
    > body-style and dimension of the car affects the rate in which cars lose their value.
    > engine specs(size, fuel-sys, num-of-cylinders, engine-type) and gas milage of that car
    > Make, Num-of-doors,body-style V. symboling and normalized-losses
    
    extra thingies (when you have time):
    - Engine-location w/ Drive-wheels and curb-weight Vs gas milage
    - Engine-location w/ Drive-wheel and curb-weight V symboling (insurance probability risk)
    - Engine type V. MPG
    - engine specs(all) and car price

    P.S: 
        - Put columns of data for inputs and outputs
        - Put test cases into new file!

    Column values of testing net: 

         Attribute:                Attribute Range:
        ------------------        -----------------------------------------------
    1. symboling:                -3, -2, -1, 0, 1, 2, 3.
    2. normalized-losses:        continuous from 65 to 256.
    3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
                                isuzu, jaguar, mazda, mercedes-benz, mercury,
                                mitsubishi, nissan, peugot, plymouth, porsche,
                                renault, saab, subaru, toyota, volkswagen, volvo
    4. fuel-type:                diesel, gas.
    5. aspiration:               std, turbo.
    6. num-of-doors:             four, two.
    7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
    8. drive-wheels:             4wd, fwd, rwd.
    9. engine-location:          front, rear.
    10. wheel-base:               continuous from 86.6 120.9.
    11. length:                   continuous from 141.1 to 208.1.
    12. width:                    continuous from 60.3 to 72.3.
    13. height:                   continuous from 47.8 to 59.8.
    14. curb-weight:              continuous from 1488 to 4066.
    15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
    16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
    17. engine-size:              continuous from 61 to 326.
    18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
    19. bore:                     continuous from 2.54 to 3.94.
    20. stroke:                   continuous from 2.07 to 4.17.
    21. compression-ratio:        continuous from 7 to 23.
    22. horsepower:               continuous from 48 to 288.
    23. peak-rpm:                 continuous from 4150 to 6600.
    24. city-mpg:                 continuous from 13 to 49.
    25. highway-mpg:              continuous from 16 to 54.
    26. price:                    continuous from 5118 to 45400.
    """
    # This is how you run a neural net! Just put some inputs, some outputs, 
    # the hidden nodes, and the location of the test cases into run_neural_net()
    # and make a neural net!
    # Arguably one of the best programs I've made.

    # Main Neurals
    # > Make, Num-of-doors,body-style V. symboling and normalized-losses
    run_neural_net([2,5,6],[0,1],10,[TEST_CASE_FILE])
    run_neural_net([0,1],[2,5,6],5,[TEST_CASE_FILE]) # Reversed

    # > engine specs(size, fuel-sys, num-of-cylinders, engine-type) and gas milage of that car

    #> body-style and dimension of the car affects the rate in which cars lose their value.

    #Extras: 
    # - Engine-location w/ Drive-wheels and curb-weight Vs gas milage
    # - Engine-location w/ Drive-wheel and curb-weight V symboling (insurance probability risk)
    # - Engine type V. MPG
    # - engine specs(all) and car price