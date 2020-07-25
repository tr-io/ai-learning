# Moore's law Linear Regression
import re
import numpy as np
import matplotlib.pyplot as plt

# load data
X = [] # the year
Y = [] # the number of transistors that year

non_decimal = re.compile(r'[^\d]+') # regex for non decimal

# relationship between log of transistor counts and year
tmap = {} # transistors map, by year

for line in open('../sample_data/moore.csv'):
    row = line.split('\t')
    year, transistors = int(non_decimal.sub('', row[2].split('[')[0])), int(non_decimal.sub('', row[1].split('[')[0]))
    if year in tmap:
        tmap[year] += transistors
    else:
        tmap[year] = transistors

for yr in tmap:
    X.append(yr)
    Y.append(np.log2(tmap[yr]))

X = np.array(X)
Y = np.array(Y)

denom = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom

Yhat = a*X + b
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("r-squared is: ", r2)

# log2(tc) = a * Year + b
# tc = exp(b) * exp(a * Year)
# 2 * tc = 2 * exp(b) * exp(a * year) = exp(ln(2)) * exp(b) * exp(a * year)
#        = exp(b) * exp(a * year + ln(2))
# exp(b) * exp(a * year2) = exp(b) * exp(a * year1 = ln2)
# a * year2 = a * year1 + ln(2)
# year2 = year + ln(2) / a

print("time to double: ", np.log2(2) / a, " years")