#!/usr/bin/env python3

# read XDATCARs from VASP and extract data like velocity, correlation function etc.
# ----------------------------
# written by Christian Braun
#    christian.braun@upb.de
# ----------------------------

from copy import deepcopy
import numpy as np
from pathlib import Path
from pdb import set_trace
from re import search

from .crystal import atomic_number, Atoms, Crystal
from .tools import direct_to_cartesian, remove_periodic, autocorrelation, first_derivative


class DynCrystal():
    # class containing a crystal with dynamic atomic positions for different time

    # _______________________________________________
    # ############### magic funcitons ###############
    # _______________________________________________
    def __init__(self, filenames, outcars=None):
        self.lattice = np.empty(9, dtype=np.float128).reshape(3, 3)  # lattice vectors
        self.elements = []  # elements in the crystal
        self.number_of_atoms = []  # number of atoms for each element defined in <self.element>
        self.direct = np.zeros((0, 0, 3), dtype=np.float)  # direct coordinates of atoms
        self.timestep = np.empty(0, dtype=np.float64)  # time steps
        self.temperature = np.empty(0, dtype=np.float64)  # temperature for each time step
        self.magnetization = np.empty(0, dtype=np.float64)  # magnetization for each time step
        self.mag_atoms = np.empty((0, 0), dtype=np.float64)  # projected magnetization for each time step
        self.borders = np.full(1, 0, dtype=np.int)  # time step where a new XDATCAR was read
        self._outcar = None  # do all XDATCARs have matching OUTCARs?

        # make sure <filenames> and <outcars> are iterabel
        filenames = filenames if isinstance(
            filenames, (list, tuple)) else (filenames, )
        outcars = [None for x in range(len(filenames))] if not outcars else (
            outcars if isinstance(outcars, (list, tuple)) else [outcars, ])
        # additionally, outcars and filenames should have equal length
        while(len(outcars) < len(filenames)):
            outcars.append(None)

        # loop through all XDATCAR & OUTCAR files
        for filename, outcar in zip(filenames, outcars):
            self.read_xdatcar(filename, outcar)

    def __len__(self):
        return len(self.position)

    def __getitem__(self, key):
        """give access to snapshot of structure at time step or slice <key>"""
        if isinstance(key, slice):
            new = self.copy()
            new.direct = new.direct[key]
            new.timestep = new.timestep[key]
            new.temperature = new.temperature[key]
            new.magnetization = new.magnetization[key]
            new.mag_atoms = new.mag_atoms[key]
            # correct borders:
            start, stop, step = key.indices(new.borders[-1])
            new.borders = np.ceil((new.borders - start)[np.where(
                np.logical_and(new.borders > start, new.borders < stop))] // step)
            new.borders = np.insert(new.borders, 0, 0.)
            if new.borders[-1] != (stop - start) // step:
                new.borders = np.append(new.borders, (stop - start) // step)
            return new
        else:
            if self._outcar:
                return self._to_crystal(self.direct[key], self.velocity[key])
            else:
                return self._to_crystal(self.direct[key])

    # __________________________________________
    # ############### properties ###############
    # __________________________________________
    @property
    def position(self):
        """
        returns the position in cartesian coordinates
        """
        return direct_to_cartesian(self.direct, self.lattice)

    @property
    def velocity(self):
        """
        calculate velocities using the finite difference formula for a 5-point stencil
        """
        # remove periodic boundary condition
        pos = direct_to_cartesian(remove_periodic(self.direct), self.lattice)

        try:
            velo = first_derivative(pos, self.timestep)
        except ZeroDivisionError:
            # velocity can only be calculated with known time step
            print('No velocity without time step!')
            return
        return velo

    @property
    def time(self):
        """
        return array of time in fs for each iteration
        """
        time = np.zeros_like(self.timestep)
        abstime = 0.0
        for i, dt in enumerate(self.timestep[1:]):
            time[i] = abstime
            abstime += dt
        time[-1] = abstime
        return time

    # _______________________________________________
    # ############### private methods ###############
    # _______________________________________________

    def _to_crystal(self, direct, velocities=None):
        """
        returns a Crystal instance with atoms having coordinates of <direct> in direct coords
        """
        # create instance of Crystal and set the positions in direct form
        crystal = Crystal(lattice=self.lattice, atoms=Atoms(
            np.sum(self.number_of_atoms), None))
        crystal.atoms.direct = direct

        # set elements
        index = [sum(self.number_of_atoms[:i]) for i in range(4)]
        for i in range(len(self.elements)):
            crystal.atoms.element[index[i]:index[i + 1]] = self.elements[i]

        # set degrees of freedom
        crystal.atoms.dof = self.dof
        if np.any(np.invert(self.dof)):
            crystal.selective_dynamics = True

        # set velocities
        if velocities is not None:
            crystal.atoms.velocity = velocities

        return crystal

    def _read_outcar(self, outcar, xdatcar):
        """
        read outcar and extract time step, temperature and magnetization
        """
        bMag = False  # do we have magnetization?
        try:
            with open(outcar, 'r') as infile:
                # check if we have spin sensitive calculations
                for line in infile:
                    if 'ISPIN' in line:
                        if int(line.strip().split()[2]) == 2:
                            bMag = True
                        break
                # read the time step
                for line in infile:
                    if 'POTIM' in line:
                        try:
                            timestep = float(search(r'POTIM  =\s*([\d\.]+)', line).groups()[0])  # time step is stored in fs!
                            break
                        except ValueError:
                            set_trace()
                # read the temperature (and magnetization)
                temperature = []
                if bMag:
                    mag = []
                    mag_atoms = []
                    for line in infile:
                        # first check for magnetization
                        if 'number of electron ' in line:
                            # magnetization is written for every electronic step
                            m = float(line.strip().split()[5])
                        elif 'magnetization (x)' in line:
                            tmp = []
                            for i in range(3):
                                infile.readline()  # skip 3 lines
                            for i in range(self.direct.shape[1]):
                                tmp.append(float(infile.readline().strip().split()[4]))
                        elif '(temperature' in line:
                            # if we are here, we have surely reached end of electronic cycle
                            mag.append(m)
                            mag_atoms.append(tmp)
                            temperature.append(float(line.strip().split()[5]))
                else:
                    for line in infile:
                        if '(temperature' in line:
                            temperature.append(float(line.strip().split()[5]))
                # check for consistency between number of timesteps in OUTCAR and XDATCAR
                if (self.direct.shape[0] == (len(self.temperature) + len(temperature))) and (self.direct.shape[0] == (len(self.magnetization) + len(mag))):
                    try:
                        self.timestep = np.append(self.timestep, np.full_like(temperature, timestep))
                        self.temperature = np.append(self.temperature, temperature)
                        self.magnetization = np.append(self.magnetization, mag)
                        self.mag_atoms = np.append(self.mag_atoms, mag_atoms, axis=0)
                    except ValueError:
                        print(outcar, xdatcar)
                        
                else:
                    print(
                        'Inconsistent number of time steps in OUTCAR and XDATCAR. OUTCAR is not read!')
                    self.timestep = np.empty(0, dtype=np.float64)
                    self.temperature = np.empty(0, dtype=np.float64)
                    self._outcar = False
                    return

        except FileNotFoundError:
            # delete all data from outcar
            print('No OUTCAR file found for {}. No temperature and time data will be available.'.format(
                xdatcar))
            self.timestep = np.empty(0, dtype=np.float64)
            self.temperature = np.empty(0, dtype=np.float64)
            return

        self._outcar = True

    # ______________________________________________
    # ############### public methods ###############
    # ______________________________________________
    def read_xdatcar(self, filename, outcar=None):
        """read xdatcar in <filename>. Looks for OUTCAR in the same directory"""
        with open(filename, 'r') as infile:
            # aquire number of configurations
            it = 0
            for line in infile:
                if 'Direct configuration=' in line:
                    it += 1
            infile.seek(0)
            
            # skip first two lines
            infile.readline()
            infile.readline()

            # read lattice vectors
            lattice = np.zeros((3, 3), dtype=np.float)
            for i in range(3):
                lattice[i] = [float(x)
                              for x in infile.readline().strip().split()]

            # read elements
            elements = infile.readline().strip().split()

            # read number of atoms
            na = np.array([int(x) for x in infile.readline().strip().split()])

            # read coordinates
            direct = np.zeros((it, na.sum(), 3), dtype=np.float)
            for i in range(it):
                infile.readline()
                for n in range(na.sum()):
                    direct[i, n] = infile.readline().strip().split()
            # numpy routine below is 50% slower!
            # direct = np.loadtxt(infile, dtype=np.float, comments='Direct')
            # direct = direct.reshape((len(direct) // np.sum(na), np.sum(na), 3))

        # check if this is first reading
        if not np.sum(self.borders):
            self.lattice = lattice
            self.elements = elements
            self.number_of_atoms = na
            self.direct = direct
            self.mag_atoms = np.empty((0, direct.shape[1]), dtype=np.float64)
        # if not, check if lattice vectors, elements and number of atoms is consistent with previous data
        elif (np.array_equal(self.lattice, lattice) and self.elements == elements and np.array_equal(self.number_of_atoms, na)):
            self.direct = np.append(self.direct, direct, axis=0)
        else:
            print(filename + ' not compatible with previous data!')
            return

        # check for degree of freedom of all atoms
        self.dof = np.invert(
            np.all(self.direct == self.direct[0, :], axis=0))

        # set new border
        self.borders = np.append(self.borders, self.borders[-1] + len(direct))

        # attempt to read outcar
        if self._outcar is not False:  # no unsuccesful readings so far
            if outcar:
                self._read_outcar(outcar, filename)
            else:
                # look for outcar in the same directory as the XDATCAR file
                self._read_outcar(
                    str(Path(filename).parent / 'OUTCAR'), filename)

    def append(self, dyn):
        """
        appends another DynCrystal to this instance
        """
        # check compatibility
        if not (np.array_equal(self.lattice, dyn.lattice) and
                np.array_equal(self.elements, dyn.elements) and
                np.array_equal(self.number_of_atoms, dyn.number_of_atoms) and
                np.array_equal(self.dof, dyn.dof)):
            print('The two DynCrystals are not compatible')
            return

        self.direct = np.append(self.direct, dyn.direct, axis=0)
        self.timestep = np.append(self.timestep, dyn.timestep, axis=0)
        self.temperature = np.append(self.temperature, dyn.temperature, axis=0)
        self.magnetization = np.append(self.magnetization, dyn.magnetization, axis=0)
        self.mag_atoms = np.append(self.mag_atoms, dyn.mag_atoms, axis=0)
        self.borders = np.append(self.borders, dyn.borders + len(self.direct), axis=0)

    def write_axsf(self, filename, stepsize=1):
        """write position of every <stepsize> iteration into an animated .xsf file"""
        it = self.direct.shape[0]  # number of animated frames
        with open(filename, 'w') as outfile:
            outfile.write('ANIMSTEPS {:d}\n'.format(it // stepsize))
            outfile.write('CRYSTAL\n')
            # write lattice vectors
            outfile.write('PRIMVEC\n')
            for i in range(3):
                outfile.write(
                    '   {:9.7f}   {:9.7f}   {:9.7f}\n'.format(*self.lattice[i]))
            # write position
            for i in range(0, it, stepsize):  # loop over time steps
                outfile.write('PRIMCOORD {:d}\n'.format(i // stepsize + 1))
                outfile.write('   {:d} 1\n'.format(self.number_of_atoms.sum()))
                start = 0
                # loop over different elements
                for j, el in enumerate(self.elements):
                    # translate element name to atomic number
                    atnum = atomic_number[el]
                    # loop over atom position of element el
                    for n in range(start, start + self.number_of_atoms[j]):
                        outfile.write('   {:3d}    {:9.7f}   {:9.7f}   {:9.7f}\n'.format(
                            atnum, *np.dot(self.lattice.T, self.direct[i, n])))
                    start += self.number_of_atoms[j]

    def copy(self):
        """returns a copy of the instance."""
        return deepcopy(self)

    def average(self, start=0, end=None):
        """returns a Crystal with the average position over all iterations ranging from <start> to <end>"""
        corrected = remove_periodic(self.direct)
        crystal = self._to_crystal(np.average(corrected[start:end], axis=0, weights=self.timestep[start:end]))
        crystal.reduce()  # now move all atoms back into the cell in case the average lies outside
        return crystal

    def std_dev(self, start=0, direct=False):
        """returns the standard deviation over all iterations starting from <start>"""
        corrected = remove_periodic(self.direct[start:])
        if not direct:
            corrected = direct_to_cartesian(corrected, self.lattice)
        return np.std(corrected, axis=0)

    def velocity_auto(self, offset=0, end=None, minSample=500):
        """compute autocorrelation function R(Dt) of the velocity from Dt=0 to Dt=<end>*dt,
dismissing the first <offset> data points and requiring at least <minSample> samples."""
        velo = self.velocity[offset:end]
        return autocorrelation(velo, minSample=minSample)

    def phonon_projection(self, phonon):
        """
        projects the velocity of the MD on the phonon eigenvectors of <phonon>.
        """
        ev = phonon.data['displacement']
        ev = ev.reshape(ev.shape[0], -1)

        velocity = self.velocity
        velocity = velocity.reshape(velocity.shape[0], ev.shape[1])
        velocity = np.moveaxis(velocity, 0, 1)

        projection = np.matmul(ev, velocity)
        return projection
