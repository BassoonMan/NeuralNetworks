def loadMatrix(filename):
    """
    Load a matrix from a text file.

    Parameters:
    filename (str): The path to the text file containing the matrix.

    Returns:
    list of list of float: The loaded matrix.
    """
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = [float(num) for num in line.split()]
            matrix.append(row)
    return matrix

def loadWords(filename):
    """
    Load words from a text file into a list.

    Parameters:
    filename (str): The path to the text file containing the words.

    Returns:
    list of str: The list of words.
    """
    words = []
    with open(filename, 'r') as file:
        for line in file:
            words.append(line.strip())
    return words
