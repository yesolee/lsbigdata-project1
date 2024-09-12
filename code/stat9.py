import numpy as np

x = np.arange(2,13)
p = np.array([1,2,3,4,5,6,5,4,3,2,1]) / 36
x_mean = sum(x*p) # 6.99
x_var = sum(((x-x_mean)**2 ) * p) # 5.83


2*x_mean+3 # 17
np.sqrt(4*x_var) # 4.83

from scipy.stats import binom

# Y~B(20,0.45)
# P(6<Y<=14)=?
sum(binom.pmf(np.arange(7,15),20,0.45))
binom.cdf(14,20,0.45) -binom.cdf(6,20,0.45)

# X~N(30,4^2)
# P(X>24) = ?
from scipy.stats import norm
1 - norm.cdf(24,30,4)

import numpy as np
from scipy.stats import chi2_contingency
phone_data = np.array([49,47,15,27,32,30]).reshape(3,2)
result = chi2_contingency(phone_data)
x_squared, p_value, df, expected = result
p_value
expected 

from scipy.stats import chisquare
import numpy as np

observed = np.array([13,23,24,20,27,18,15])
expected = np.repeat(20,7)
statistic, p_value = chisquare(observed, f_exp=expected)

statistic
p_value 