import numpy as np
# -2/5*log2(2/5) -3/5*log2(3/5)
p_r = 2/5
p_b = 3/5
h_zero = -p_r*np.log2(p_r) - p_b*np.log2(p_b)
h_zero

p_r = 1/4
p_b= 3/4
h_1_r = -p_r*np.log2(p_r) - p_b*np.log2(p_b)
h_1_r*4/5-h_zero

