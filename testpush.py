#this is a file that test push github from other computer
import numpy as np
a = np.ones((64,256))
b = np.ones((64,256))

c = np.concatenate((a,b),axis=0)
d = 1
print(c.shape)

