import numpy as np
from sympy import Derivative

#Task 1A 
#--- Derivative calculation---
def derivative(f, x, h=1e-4): 
    return (f(x + h) - f(x)) / h

def f(x):
    return 4*((x-2)**4)

#---Newton method---
def newton_method(init_guess, f, tolerance=1e-6, max_iterations=100):
    x = init_guess

    for iteration in range(max_iterations):
        # First derivative
        f_prime_x = derivative(f, x)

        # Second Derivative
        f_double_prime_x = derivative(lambda x: derivative(f, x), x)

        if f_double_prime_x == 0:
            #print("can't divide by zero")
            return None
        
        #---Newton Method---
        delta_x = - f_prime_x / f_double_prime_x

        #---x update---
        x += delta_x
        # print(f"{x} \n ") # Commented out so it doesn't print 1000 times
        
        if abs(delta_x) < tolerance:
            #print("Converged after", iteration + 1, "iterations.")
            return x

    #print("Did not converge within", max_iterations, "iterations.")
    return None


newton_method(21,f)

#Task 1B

extrems=[]
def f1(x):
    return (2*x**4 + x**3 - 3*x**2 + 5)


def mult_newt(f,num_guesses,init_guess_range=[], ):
    init_guesses=np.linspace(init_guess_range[0],init_guess_range[1],num_guesses )
    for i in range(num_guesses):
        t=newton_method(init_guesses[i],f)
        if t is not None:
            extrems.append(t)
    
    
    return np.unique(np.round(extrems, 4))
if __name__ == "___main___": 
    e=mult_newt(f1,1000,[-2,2])
    print(e)

    for i in range(len(e)):
        print(f"value of extrema {1}: {f1(e[i])} \n")