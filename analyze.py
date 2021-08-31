#!/usr/bin/env python3

import numpy as np
from .parameters import atomic_structure_coeffecient
from .tools import direct_to_cartesian, cartesian_to_direct, remove_periodic


def structure_factor(crystal, G):
    """calculate structure factor of crystal for reciprocal vector G"""
    # define function for atomic factor
    def atomic_struc(element):
        """calculate atomic structure factor for element el and reciprocal vector G in direct coordinates and rec_lattice in A^-1"""
        # calculate scalar reciprocal wave number
        q = np.linalg.norm(np.tensordot(G, crystal.reciprocal_lattice, axes=([0], [0])))
        asc = np.array(atomic_structure_coeffecient[element])
        return np.dot(asc[0:8:2],
                      np.exp(-(q / (4 * np.pi))**2 * asc[1:8:2])) + asc[8]

    # calculate structure factor element-wise
    sf = 0
    for element in crystal.elements:
        # calculate reciprocal lattice vectors
        f = atomic_struc(element)
        sf += f * np.sum(np.exp(-2 * np.pi * 1j * np.tensordot(crystal.atoms[element].direct.copy(), G, axes=([1], [0]))))
    return sf


def dynamic_structure_factor(dyncrystal, G):
    """calculate the atomic structure factor incl. the Debye-Waller factor for element el
and reciprocal vector G in direct coordinates and rec_lattice in A^-1"""
    # get the average structure
    crystal = dyncrystal.average()

    # define function for atomic factor
    def atomic_struc(element):
        """calculate atomic structure factor for element el and reciprocal vector G in direct coordinates and rec_lattice in A^-1"""
        # calculate scalar reciprocal wave number
        q = np.linalg.norm(np.tensordot(G, crystal.reciprocal_lattice, axes=([0], [0])))
        asc = np.array(atomic_structure_coeffecient[element])
        return np.dot(asc[0:8:2],
                      np.exp(-(q / (4 * np.pi))**2 * asc[1:8:2])) + asc[8]

    p = np.average(np.exp(2 * np.pi * 1j * np.dot(dyncrystal.direct, G)), axis=0)
    f = [atomic_struc(crystal.atoms[i].element) for i in range(crystal.number_all_atoms)]

    return np.sum(p * f)
