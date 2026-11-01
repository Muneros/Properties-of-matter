from set3 import  mult_newt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#1.1
def f1(x):
    return x**2 + 15*np.sin(x)
vals=[]
x=np.linspace(-10,10,1000)
for i in range(len(x)):
    vals.append(f1(x[i]))


plt.scatter(x,vals)
plt.show()


#1.2
#loval minima/maxima are around : x = -6.7 (Local Minimum) 
# x = -5.5 (Local Maximum) 
# x = -1.4 (Local Minimum)
#x = 1.8 (Local Maximum)
#x = 4.2 (Local Minimum)

#--- Checking values using method from set 3 ---
e=mult_newt(f1,1000,[-10,10])
print(f"X of extrema: {e}")
print(f"Y of extrema {f1(e)}")

min1_x = minimize(f1, -6.7, method='L-BFGS-B', bounds=[(-7, -5)]).x[0]
min2_x = minimize(f1, -1.4, method='L-BFGS-B', bounds=[(-2, -1)]).x[0]
min3_x = minimize(f1, 4.2, method='L-BFGS-B', bounds=[(3, 5)]).x[0] # Fixed: passed 4.2 instead of -1.4

minima_x = np.array([min1_x, min2_x, min3_x])

plt.scatter(x, vals, s=1)
#ymin = -20 so the middle minimum is getting drawn
plt.vlines(x=minima_x, ymin=-20, ymax=120, color="red", linestyle='--', zorder=5)

plt.show()


#1.3

def g(t):
    x, y = t[0], t[1]
  
    term1 = 2 * np.exp(-x**2 - y**2)
    term2 = -np.exp(-(x - 1)**2 - (y - 1)**2)
    term3 = -2 * np.exp(-(x - 3)**2 - y**2) 
    return term1 + term2 + term3


x = np.linspace(-4, 5, 400)
y = np.linspace(-4, 5, 400)
X, Y = np.meshgrid(x, y)
Z = g([X, Y])

plt.figure(figsize=(7, 6))


contours = plt.contour(X, Y, Z, levels=12, cmap='viridis')


plt.clabel(contours, inline=True, fontsize=9, fmt='%.2f')


plt.colorbar(contours)


plt.plot(0, 0, marker='o', color='red', markersize=7)
plt.plot(1, 1, marker='o', color='blue', markersize=7)
plt.plot(3, -2, marker='o', color='red', markersize=7) 

plt.title('Contour plot g(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-4, 5)
plt.ylim(-4, 5)

plt.show()