#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import crystal as cst
from crystal import md


def plot_temperature(dynCrystal, start=0, end=None, time=False):
    """Plot temperature with respect to time"""
    if dynCrystal.temperature.size:
        dt = dynCrystal.timestep
        if time:
            x = np.arange(start * dt, (end or dynCrystal.temperature.shape[0]) * dt, dt)
        else:
            x = np.arange(start, (end or dynCrystal.temperature.shape[0]))
        y = dynCrystal.temperature[start:]
        plt.plot(x, y)
        plt.show()


def plot_velocity_auto(dynCrystal, offset=0, end=None, minSample=400, time=False):
    """Plot temperature with respect to time"""
    plt.close()
    if dynCrystal.timestep:
        dt = dynCrystal.timestep
        y = dynCrystal.velocity_auto(offset, end, minSample)
        if time:
            x = np.arange(0.0, (end or y.shape[0]) * dt, dt)
        else:
            x = np.arange(0.0, (end or y.shape[0]))
        plt.plot(x, y)
        plt.show()
