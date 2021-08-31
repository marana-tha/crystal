#!/usr/bin/env python3

import numpy as np
from pdb import set_trace


def direct_to_cartesian(direct, lattice):
    """transforms direct coordinates to cartesian coordinates"""
    return np.tensordot(direct, lattice, axes=(-1, 0))


def cartesian_to_direct(cartesian, lattice):
    """transforms cartesian coordinates to direct coordinates"""
    return np.tensordot(cartesian, np.linalg.inv(lattice), axes=(-1, 0))


def remove_periodic(coordinates, direct=True, lattice=None):
    """removes the periodic boundaries of a set of coordinates so that atoms are not "beamed" to the other side of the cell during time evolution.
For <direct> = True, coordinates in direct form are expected and returned. Otherwise cartesian coord. are expected and returned and <lattice> must be set."""
    coords = coordinates if direct else cartesian_to_direct(coordinates, lattice)  # work with direct coordinates
    distance = coords - coords[0]
    corrected = np.where(np.abs(distance) > 0.5,  # atoms which have moved more than half the length of cell must have "beamed"
                         np.where(distance > 0.5, coords - 1.0, coords + 1.0),  # distinguish between direction of "beaming"
                         coords)
    return corrected if direct else direct_to_cartesian(corrected, lattice)


def autocorrelation(data, minSample=500):
    """compute autocorrelation function of data, averaging over at least <minSample> data points.
The first axis of data must be the time axis."""
    # size of autocorrelation function
    npoints = np.shape(data)[0] - minSample
    try:
        auto = np.zeros(shape=(npoints, *data.shape[1:]), dtype=np.float64)
    except ValueError:
        print("Not enough data points for the required number of samples")
        return

    # compute autocorrelation function
    auto[0] = np.sum(data**2, axis=0) / data.shape[0]
    for i in range(1, npoints):
        auto[i] = np.sum(data[:-i] * data[i:], axis=0) / (data.shape[0] - i)

    return auto


def first_derivative(data, stepsize):
    """Calculate the first derivative of the first axis of data.
<stepsize> can be an array with stepsize[i] containing the step size between data[i-1] and data[i]."""
    # allocate memory
    deriv = np.zeros_like(data)
    if hasattr(stepsize, '__iter__'):
        dt = stepsize
    else:
        dt = np.full(len(data), stepsize)

    # make dt and data broadcastable
    while (dt.ndim < data.ndim):
        dt = np.expand_dims(dt, axis=-1)

    # treat borders separatly
    deriv[0] = -(dt[1] / (dt[2] * (dt[1] + dt[2]))) * data[2] + ((dt[1] + dt[2]) / (dt[1] * dt[2])) * data[1] - ((2 * dt[1] + dt[2]) / (dt[1] * (dt[1] + dt[2]))) * data[0]
    deriv[1] = (data[2] + (dt[2]**2 / dt[1]**2) * (data[1] - data[0]) - data[1]) * (dt[1] / (dt[2] * dt[1] + dt[2]**2))
    deriv[-2] = (data[-1] + (dt[-1]**2 / dt[-2]**2) * (data[-2] - data[-3]) - data[-2]) * (dt[-2] / (dt[-1] * dt[-2] + dt[-1]**2))
    deriv[-1] = ((2 * dt[-1] + dt[-2]) / (dt[-1] * (dt[-1] + dt[-2]))) * data[-1] - ((dt[-1] + dt[-2]) / (dt[-1] * dt[-2])) * data[-2] + (dt[-1] / (dt[-2] * (dt[-1] + dt[-2]))) * data[-3]
    # treat borders separatly
    # deriv[0] = (-1.0 * data[2] + 4.0 * data[1] - 3.0 * data[0]) / (2.0 * dt[1])  # forward difference
    # deriv[1] = (data[2] - data[0]) / (dt[1] + dt[2])  # central difference
    # deriv[-2] = (data[-1] - data[-3]) / (dt[-2] + dt[-1])  # central difference
    # deriv[-1] = (3.0 * data[-1] - 4.0 * data[-2] + 1.0 * data[-3]) / (2.0 * dt[-1])  # backward difference

    # for remaining derivatives use 5-point stencil with fexible time steps
    # define single forward and backward and double forward and backward step
    fdt = dt[3:-1]  # one step in forward direction
    bdt = dt[2:-2]  # one step in backward direction
    fdt2 = (dt[3:-1] + dt[4:]) / 2.0  # two steps in forward direction
    bdt2 = (dt[1:-3] + dt[2:-2]) / 2.0  # two steps in backward direction
    # calculate coefficients for flexible 5-point stencil
    k = 8 * (fdt2**2 * (fdt2 + bdt2)) / (fdt**2 * (fdt + bdt))
    z = fdt2**2 * (fdt2 + bdt2) * (8.0 * (bdt - fdt) / (fdt**2 * bdt**2) - (bdt2 - fdt2) / (fdt2**2 * bdt2**2))
    n = fdt2**2 * (fdt2 + bdt2) * (8.0 / (fdt * bdt) - 2.0 / (fdt2 * bdt2))
    # df/di = ( -f(i+2) + k * f(i+1) - z * f(i) - k * f(i-1) + f(i-2) ) / l
    deriv[2:-2] = (-data[4:] + k * data[3:-1] - z * data[2:-2] - k * fdt**2 / bdt**2 * data[1:-3] + fdt2**2 / bdt2**2 * data[:-4]) / n

    return deriv


def list_to_slice(items):
    # tries to transform a list of positiv integers into a slice object
    try:
        step = items[1] - items[0]
        # increment between consecutive elements must be constant
        if np.all((items[1:] - items[:-1]) == step):
            return slice(np.amin(items), np.amax(items) + 1, step)
    except IndexError:
        pass
    return items


def divisorGenerator(n):
    small_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield n // i
            if i * i != n:
                small_divisors.append(i)
    for divisor in reversed(small_divisors):
        yield divisor


def distance_along_line(crystal, point, i):
    """
    return the distance of all atoms in <crystal> from a line corresponding to
    lattice vector <i> passing through point <point>.
    """
    point -= np.floor(point)
    distance = crystal.atoms.direct - point
    distance[:, i] = 0  # no need in considering distances along direction <i>

    # take periodic boundaries into consideration
    distance -= np.around(distance)

    cartDist = np.linalg.norm(direct_to_cartesian(distance, crystal.lattice), axis=1)  # calculate the distance in cartesian coordinates
    return cartDist
