import numpy


def db_to_linear(x):
    return 10 ** (x/10)


def linear_to_db(x):
    return 10 * numpy.log10(x)
