#float(a)


'''

a = 1.5 + 0.5j 
a.real >>> 1.5
a.imag >>> 0.5

complex(re,im) where re is the real part and im is imaginary part

1. Because 7 and three are integers
2. By writing 7/3.
3. Use //

2.1 Only prints every three characters.
Prints string in reverse order.
string[::-1]


'''
#P3
my_list = ["mushrooms", "rock climbing", 1947, 1954, "yoga"]
#a
len(my_list)
my_list.append("Jonathan, my pet fish")
my_list.insert(3, "pizza")
del my_list[:]
#2
num = []
num.extend([3, 5, 19, 20, 4])
num[3] = str(num[3])
num[::-1]
num = [str(x) for x in num]
#curly braces or function set()
#set()
