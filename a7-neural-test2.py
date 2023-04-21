from typing import Tuple, List

from sklearn import train_test_split

DATA_FILE = "testing_data/imports-85.data"


def parse_line(line: str, out_put: int) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    print(tokens)
    try:
        out = int(tokens[0])
    except ValueError:
        pass
    
    inpt = [float(x) for x in tokens[1:] if x.isnumeric()]
    out = int(tokens[out_put])
    output = [0 if out == 1 else 0.5 if out == 2 else 1]

    return (inpt, output)


# Imported from neural_net data
def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            print(data[i][0][j])
            if type(data[i][0][j]) == str:
                break
            else:
                if data[i][0][j] < leasts[j]:
                    leasts[j] = data[i][0][j]
                if data[i][0][j] > mosts[j]:
                    mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
           if type(data[i][0][j]) == str:
                break
           else:
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

def convert_strs(data: List[Tuple[List[float], List[float]]]):
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            pass
            

if __name__ == "__main__":
    with open(DATA_FILE, "r") as f:
        training_data = [parse_line(line,23) for line in f.readlines() if len(line) > 4]
    print(training_data)
    td = normalize(training_data)
    print(td)

    # nn = nnd.NeuralNet(13, 3, 1)
    # nn.train(td) # , iters=100_000, print_interval=1000, learning_rate=0.1)

    # for i in nn.test_with_expected(td):
    #     print(f"desired: {i[1]}, actual: {i[2]}")
