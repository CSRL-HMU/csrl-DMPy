import math

# 3.14159...
pi = math.pi

# the sinc function
def sinc(x):
    if x == 0: 
        return 1.0
    else:
        return math.sin(x)/x 
        


