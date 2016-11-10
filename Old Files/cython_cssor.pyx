from numpy cimport ndarray as ar
cdef extern from "cssor.h":
	void cssor(double* U, int m, int n, double omega, double tol, int maxiters, int* info)

cpdef cyssor(ar[double, ndim=2] U, double omega=1.9, double tol=1e-9, maxiters=10000):
	cdef int m,n,info
	if U.flags['C_CONTIGUOUS']:
		m=U.shape[1]
		n=U.shape[0]
		cssor(&U[0,0],m,n,omega,tol,maxiters,&info)
	else:
		raise ValueError("Input array U is not C-contiguous")
	if info==1:
		raise ValueError("You broke math, this did not converge!")
