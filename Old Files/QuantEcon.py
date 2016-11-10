import timeit
import Lab15v1

#1
#before

def linapprox(f,a,b,n,x):
	interval_length=b-a
	subintervals=n-1
	steps= interval_length/subintervals
	point=a
	while point <= x:
		point +=steps
	t, s=point-steps, point
	return f(t)+(x-t)*(f(s)-f(t))/(s-t)

#after
def linapproxr(f,a,b,n,x):
	steps= (b-a)/(n-1)
	point=a
	while point <= x:
		point +=steps
	t, s=point-steps, point
	return f(t)+(x-t)*(f(s)-f(t))/(s-t)



#I cut out specific variables and simply inserted their values into the spots where they were in the first function




def time_func(f, args =(), repeat = 3, number = 100):
    pfunc = lambda: f(*args)
    T = timeit.Timer(pfunc)
    try:
        t = T.repeat(repeat=repeat, number=int(number))
        runtime = min(t)/float(number)
        return runtime
    except:
        T.print_exc()

f =lambda x:x**2

for i in range(10):
    print time_func(linapprox, (f,0,1,2,0.5))
for i in range(10):
    print time_func(linapproxr, (f,0,1,2,0.5))



#2

X = [1, 2, 3]

print Lab15v1.pysum(X)






