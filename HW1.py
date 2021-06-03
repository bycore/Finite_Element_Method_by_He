import Gaussian_integral
import base_function
import numpy as np
import sympy
import math

def fun_f(x):
    # return 1
    return math.exp(x)

def fun_g(x):
    # return -2.0+0*x
    return -1*sympy.exp(x)*(sympy.cos(x)-2*sympy.sin(x)-x*sympy.cos(x)-x*sympy.sin(x))

N=16
a,b=0.0,1.0
h=(b-a)/N

P=np.linspace(a,b,N+1)
temp=np.linspace(0,N,N+1,dtype=int)
T=np.vstack([temp[:-1],temp[1:]])
base_der=np.eye(N+1,N,dtype=float)*(-1/h)+np.eye(N+1,N,k=-1,dtype=float)*1/h

x=sympy.symbols("x")

A=np.zeros((N+1,N+1))

for n in range(N):
    for i in range(N+1):
        for j in range(N+1):
            A[i][j]=A[i][j]+base_der[j][n]*base_der[i][n]*Gaussian_integral.Gau_3(fun_f,P[T[0][n]],P[T[1][n]])


b=np.zeros((N+1,1))
for i in range(N+1):
    for n in range(N):
        # print(P[T[0][n]],P[T[1][n]],base_function.hat_function(i,n,x,P,T))
        b[i]+=float(Gaussian_integral.Gau_sym_3(fun_g(x)*base_function.hat_function(i,n,x,P,T),P[T[0][n]],P[T[1][n]]))
        # b[i]=b[i]+base_der[i][n]*Gaussian_integral.Gau_3(fun_g,P[T[0][n]],P[T[1][n]])

A[0,:]=0
A[0][0]=1
A[-1,:]=0
A[-1][-1]=1
b[0]=0
b[-1]=math.cos(1)

# print(A)
# print(b)

anx = np.linalg.solve(A,b)
print(anx)
err=-1
j=0
for i in anx:
    err=max(i-P[j]*math.cos(P[j]),err)
    j=j+1
print(err)




