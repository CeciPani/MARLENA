import numpy as np
from scipy.spatial import distance


def minmax_dist(x, y):
    x, y = np.array(x), np.array(y)
    return abs((x[0] - y[0]))/ np.concatenate((x, y)).max()

def normalized_euclidean_distance(x, y):
    if len(x)>1:
        return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))
    else:
        return minmax_dist(x,y)

def simple_match_distance(x,y):
    return distance.hamming(x,y)/len(x)


#def simple_match_distance(x, y):
#    count = 0
#    for xi, yi in zip(x, y):
#        if xi == yi:
#            count += 1
#    sim_ratio = 1.0 * count / len(x)
#    return 1.0 - sim_ratio


def normalized_square_euclidean_distance(ranges):
    def actual(x, y, xy_ranges):
        return np.sum(np.square(np.abs(x - y) / xy_ranges))

    return lambda x, y: actual(x, y, ranges)


def mad_distance(x, y, mad):
    val = 0.0
    for i in range(len(mad)):
        val += 0.0 if mad[i] == 0.0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
    return val


def mixed_distance_series(x, y, categorical, numerical, labels_name, cat_dist, num_dist):
    # type (pandas.Series, pandas.Series, list, list, list, function, function) -> double
    """
    This function return the mixed distance between instance x and instance y
    :param x: pandas.Series, instance 1
    :param y: pandas.Series, instance 2
    :param categorical: list of str, column names containing categorical variables
    :param numerical: list of str, column names containing non categorical variables
    :param labels_name: list of str, array of column names containing the label
    :param cat_dist: function, distance function for categorical variables
    :param num_dist: function, distance function for continuos variables
    :return: double
    """
    print(x)
    print(categorical)
    xd = [x[att] for att in categorical if att not in labels_name]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in categorical if att not in labels_name]
        wd = len(categorical) / (len(categorical) + len(numerical))
        dd = cat_dist(xd, yd)

    xc = np.array([x[att] for att in numerical if att not in labels_name])
    wc = 0.0
    cd = 0.0
    if len(xc) > 0:
        yc = np.array([y[att] for att in numerical if att not in labels_name])
        wc = len(numerical) / (len(categorical) + len(numerical))
        cd = num_dist(xc, yc)

    return wd * dd + wc * cd


def mixed_distance_arrays(x, y, len_categorical, len_numerical, cat_dist, num_dist):
    # type (numpy.Array, numpy.Array, int, int, function, function) -> double
    """
    This function return the mixed distance between instance x and instance y
    
    N.B.: the instances values should be sorted in the following way: first the 'len_categorical' values and then the 'len_numerical' values
    
    :param x: numpy.Array, instance 1
    :param y: numpy.Array, instance 2
    :param len_categorical: int, the number of categorical 
    :param len_numerical: int, column names containing non categorical variables
    :param cat_dist: function, distance function for categorical variables
    :param num_dist: function, distance function for numerical variables
    :return: double
    """
    #print(f'x = {x}')
    xd = x[:len_categorical]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = y[:len_categorical]
        wd = len_categorical / (len_categorical + len_numerical)
        dd = cat_dist(xd, yd)

    xc = x[len_categorical: len_categorical + len_numerical]
    #print(f'xc = {xc}')
    wc = 0.0
    cd = 0.0
    if len(xc) > 0:
        yc = y[len_categorical: len_categorical + len_numerical]
        wc = len_numerical / (len_categorical + len_numerical)
        cd = num_dist(xc, yc)

    return wd * dd + wc * cd
