# name this file 'solutions.py'
"""Volume II Lab 1: The Standard Library
Chris Rytting
Math 321
9/3/15
<Class>
<Date>
"""

import calculator as calc
# In future labs, do not modify any PROVIDED import statements.
# You may always add others as needed.


# Problem 1: Implement this function.
def prob1(numbers):
    maximum = min(numbers)
    minimum = max(numbers)
    average = float(sum(numbers))/len(numbers)
    return [maximum, minimum, average]
    pass
A = [1,2,3,4]

# Problem 2: Implement this function.
def prob2():
    #numbers
    number_1 = 1
    number_2 = number_1
    number_1 += 1
    print(number_1)
    print(number_2)
    #strings
    string_1 = 'Chris'
    string_2 = string_1
    string_1 += ' Rytting'
    print(string_1)
    print(string_2)
    #lists
    list_1 = [1,2,3]
    list_2 = list_1
    list_1.append(4)
    print(list_1)
    print(list_2)
    #tuples
    tuple_1 = (1,2,3)
    tuple_2 = tuple_1
    tuple_1 = tuple_1 + (4,)
    print(tuple_1)
    print(tuple_2)
    #dictionaries
    dictionary_1 = {'jack': 'a man who attacks'}
    dictionary_2 = dictionary_1
    dictionary_1['maurice'] = 'A woman man'
    print(dictionary_1)
    print(dictionary_2)
    pass


# Problem 3: Create a 'calculator' module and use it to implement this function.
def prob3(a,b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any methods other than those that are imported from the
    'calculator' module.
    
    Parameters:
        a (float): the length one of the sides of the triangle.
        b (float): the length the other nonhypotenuse side of the triangle.
    
    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.square_root_of_number(calc.add_numbers(calc.multiply_numbers(a,a), calc.multiply_numbers(b,b)))
    pass
a = 3
b = 4



# Problem 4: Utilize the 'matrix_multiply' module and 'matrices.npz' file to
#   implement this function.
def prob4():
    """If no command line argument is given, print "No Input."
    If anything other than "matrices.npz is given, print "Incorrect Input."
    If "matrices.npz" is given as a command line argument, use functions
    from the provided 'matrix_multiply' module to load two matrices, then
    time how long each method takes to multiply the two matrices together.
    Print your results to the terminal.
    """
    import matrix_multiply as mm
    import sys
    import time
    if len(sys.argv) < 2:
        print 'No Input'
    elif len(sys.argv) == 2:
        if sys.argv[1] != 'matrices.npz':
            print 'Incorrect input'
        if sys.argv[1] == 'matrices.npz':
            A,B = mm.load_matrices(sys.argv[1])
            start = time.time()
            mm.method1(A,B)
            end = time.time()
            print "Time for method {}".format(1), abs(start - end)
            start = time.time()
            mm.method2(A,B)
            end = time.time()
            print "Time for method {}".format(2), abs(start - end)
            start = time.time()
            mm.method3(A,B)
            end = time.time()
            print "Time for method {}".format(3), abs(start - end)
    else:
        pass
    pass


# Everything under this 'if' statement is executed when this file is run from
#   the terminal. In this case, if we enter 'python solutions.py word' into
#   the terminal, then sys.argv is ['solutions.py', 'word'], and prob4() is
#   executed. Note that the arguments are parsed as strings. Do not modify.
if __name__ == "__main__":
    prob4()


# ============================== END OF FILE ================================ #
