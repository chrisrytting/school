print "Quant Econ"
print "Exercise 1"
print "Part 1:"
x_vals=[1,4,6,7]
y_vals=[3,8,1,2]
print sum([x*y for x, y in zip(x_vals, y_vals)])

print "Part 2:"
print sum([i%2==0 for i in range(100)])
print "Part 3:"
pairs= ((2,5), (4,2), (9,8), (12,10))
print sum([x%2==0 and y%2==0 for x, y in pairs])
print "Exercise 2"

def p(x, coeff):
	print sum(a*x**n for n, a in enumerate(coeff))
p(3,(2,6))

print "Exercise 3"

def capitalize(string):
    counter = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            counter += 1
    return counter
print capitalize('We argue A lot')

print "Exercise 4"

def truth(seq_a, seq_b):
	is_subset=True
	for i in seq_a:
		if i not in seq_b:
			is_subset=False
 	return is_subset
print (truth([2],[3,4]))
print(truth([2],[2,6]))

print "problem 5:"

def linapprox(f,a,b,n,x):
	interval_length=b-a
	subintervals=n-1
	steps= interval_length/subintervals
	point=a
	while point <= x:
		point +=steps
	t, s=point-steps, point
	return f(t)+(x-t)*(f(s)-f(t))/(s-t)

f =lambda x:x**2
print linapprox(f,0,1,2,0.5)




		
