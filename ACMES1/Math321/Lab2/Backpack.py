# Backpack.py
"""Volume II Lab 2: Object Oriented Programming (Auxiliary file)
Modify this file for problems 1 and 3.
Chris Rytting
Math 321
9/24/15
"""

# Problem 1: Modify this class. Add 'name' and max_size' attributes, modify
#   the put() method, and add a dump() method. Remember to update docstrings.
class Backpack:
    def __init__(self, color = 'black', name = 'backpack', max_size = 5):
        """Constructor for a backpack object.
        Set the color, the name, the max_size and initialize the contents list.
        Inputs:
        color (str, opt): the color of the backpack. Defaults to 'black'.
        name (str, opt): the name of the backpack. Defaults to 'backpack'.
        max_size (int, opt): the maximum size of the backpack. Defaults to 5.
        Returns:
            A backpack object wth no contents.
        """
        # Assign the backpack a color.
        self.color = color
        self.equal = False
        # Assign the backpack a name.
        self.name = name
        # Assign the backpack a maximum size
        self.max_size = max_size
        # Create a list to store the contents of the backpack.
        self.contents = []
    def put(self, item):
        """Add 'item' to the backpack's content list.""" 
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        elif len(self.contents) >= self.max_size:
            print "Backpack Full."
    def take(self, item):
        """Remove 'item' from the backpack's content list.""" 
        self.contents.remove(item)
    def dump(self):
        """Remove 'item' from the backpack's content list.""" 
        self.contents[:] = []
    def __str__(self):
        """Convert information about backpack to a string"""
        string = ''
        contentsstring = ''
        if self.contents == []:
            contentsstring = 'Empty'
        else:
            for i in self.contents:
                contentsstring +='\n          '
                contentsstring += str(i)
        string += 'Name:     ' + str(self.name) + '\nColor:    ' + str(self.color) + '\nSize:     ' + str(len(self.contents)) + '\nMax Size: ' + str(self.max_size) + '\nContents: ' + contentsstring
        return string
    def __eq__(self, other):
        """Check if two backpacks are the same"""
        contentsmatchfail = False
        equal = False
        for i in self.contents:
            if i in other.contents:
                pass
            else:
                contentsmatchfail = True
        for i in other.contents:
            if i in self.contents:
                pass
            else:
                contentsmatchfail = True
        if self.name == other.name and self.name == other.name and contentsmatchfail == False:
            equal = True
        return equal

# Study this example of inheritance. You are not required to modify it.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.
    
    Attributes:
        color (str): the color of the knapsack.
        name (str): the name of the knapsack.
        max_size (int): the maximum number of items that can fit in the
            knapsack.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    
    def __init__(self, color='brown', name='knapsack', max_size=3):
        """Constructor for a knapsack object. A knapsack only holds 3 item by
        default instead of 5. Use the Backpack constructor to initialize the
        name and max_size attributes.
        
        Inputs:
            color (str, opt): the color of the knapsack. Defaults to 'brown'.
            name (str, opt): the name of the knapsack. Defaults to 'knapsack'.
            max_size (int, opt): the maximum number of items that can be
                stored in the knapsack. Defaults to 3.
        
        Returns:
            A knapsack object with no contents.
        """
        
        Backpack.__init__(self, color, name, max_size)
        self.closed = True
    
    def put(self, item):
        """If the knapsack is untied, use the Backpack put() method."""
        if self.closed:
            print "Knapsack closed!"
        else:
            Backpack.put(self, item)
    
    def take(self, item):
        """If the knapsack is untied, use the Backpack take() method."""
        if self.closed:
            print "Knapsack closed!"
        else:
            Backpack.take(self, item)
    
    def untie(self):
        """Untie the knapsack."""
        self.closed = False
    
    def tie(self):
        """Tie the knapsack."""
        self.closed = True

# ============================== END OF FILE ================================ #
