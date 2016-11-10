# solutions.py
"""Volume I Lab 1: Getting Started
Chris Rytting
Math 345
Sep 4 2015
"""
import string

# Problem 1: Write and run a "Hello World" script.
print 'Hello World'

# Problem 2: Implement this function.
def sphere_volume(r):
    """Return the volume of the sphere of radius 'r'."""
    pi = 3.14159
    return (4./3)*pi*r**3
    pass

# Problem 3: Implement the first_half() and reverse() functions.
def first_half(my_string):
    """Return the first half of the string 'my_string'.

    Example:
        >>> first_half("python")
        'pyt'
    """
    length = len(my_string)
    return my_string[:length/2]
    pass
    
def reverse(my_string):
    """Return the reverse of the string 'my_string'.
    
    Example:
        >>> reverse("python")
        'nohtyp'
    """
    return my_string[::-1]
    pass

    
# Problem 4: Perform list operations
# For the grader, do not change the name of 'my_list'.
my_list =  ["ant", "baboon", "cat", "dog"] 

# Put your code here
my_list.append('elephant')
my_list.remove('ant')
my_list.pop(1)
print my_list
my_list.pop(2)
my_list.insert(2,'donkey')
my_list.append('fox')
print my_list
    
# Problem 5: Implement this function.
def pig_latin(word):
    """Translate the string 'word' into Pig Latin
    
    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    
    if word[0] in (vowels):
        word += 'hay'
    else:
        c = word[:1]
        word = word[1:]
        word += c
        word += 'ay'
    return word
    pass
print pig_latin('banana')

        
# Problem 6: Implement this function.
def int_to_string(my_list):
    """Use a dictionary to translate a list of numbers 1-26 to corresponding
    lowercase letters of the alphabet. 1 -> a, 2 -> b, 3 -> c, and so on.
    
    Example:
        >>> int_to_string([13, 1, 20, 8])
        ['m', 'a', 't', 'h'] 
    """
    dictionary = dict()
    for index, letter in enumerate(string.ascii_lowercase):
       dictionary[index + 1] = letter
    strret = []
    for i in my_list:
        strret.append(dictionary[i])
    return strret



    
    pass

# Problem 7: Implement this generator.
def squares(n):
    """Yield all squares less than 'n'.

    Example:
        >>> for i in squares(10):
        ...     print(i)
        ... 
        0
        1
        4
        9
    """
    i = 0
    while i**2 < n:
        yield i**2
        i+=1
    pass


# Problem 8: Implement this function.
def stringify(my_list):
    """Using a list comprehension, convert the list of integers 'my_list'
    to a list of strings. Return the new list.

    Example:
        >>> stringify([1, 2, 3])
        ['1', '2', '3']
    """
    strret = list()
    for n in xrange(len(my_list)):
        strret.append(str(my_list[n]))
    return strret
    pass

# Problem 9: Implement this function and use it to approximate ln(2).
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximae ln(2).
    """
    total = 0
    for i in xrange(1,n+1):
        if i % 2 == 1:
            total += 1./i
        if i % 2 == 0:
            total -= 1./i
    return total
    pass

ln2 = alt_harmonic(5000) # put your approximation for ln(2) here
print ln2
