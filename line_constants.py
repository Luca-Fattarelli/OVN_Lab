import numpy
# gain and figure in dB
gain = 16
noise_figure = 3
# f_ase -> frequency for ase generation
f_ase = 193.414e12
# noise bandwidth
b_n = 12.5
# es 3 lab 7
# db / m
alfa_db = 0.2e-3
alfa = alfa_db / (10 * numpy.log10(numpy.e))
l_eff = 1 / alfa
# m * Hz^2
beta_2 = 2.13e-26
# (m W) ^ -1
gamma = 1.27e-3
# Ghz
r_s = 32
df = 50
