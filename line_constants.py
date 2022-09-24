import numpy
# gain and figure in dB
gain = 16
noise_figure = 3
# noise_figure = 5
# f
f = 193.414e12
# noise bandwidth
b_n = 12.5e9
# es 3 lab 7
# db / m
alfa_db = 0.2e-3
alfa = alfa_db / (10 * numpy.log10(numpy.e))
l_eff = 1 / alfa
# m * Hz^2
beta_2 = 2.13e-26
# beta_2 = 0.6e-26
# (m W) ^ -1
gamma = 1.27e-3
# Ghz
r_s = 32e9
df = 50e9
