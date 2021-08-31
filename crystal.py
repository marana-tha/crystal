#!/usr/local/bin/python3

# read and manipulate POSCAR files
# ----------------------------
# written by Christian Braun
#    braun91@mail.upb.de
# ----------------------------

import copy
from functools import wraps
import inspect
import numpy as np
import os
from os.path import dirname
import re
from threading import Thread

from .tools import list_to_slice, direct_to_cartesian, cartesian_to_direct, divisorGenerator

# dictionary to translate abbrev. to coordinate numbers
atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
                 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
                 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
                 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
                 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
                 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113,
                 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118}


class selectable():
    """decorator class, intercepting several arguments given to the decorated function. It compiles a list of
indices, for which these arguments match.

    All attributes of <class Atom> can be used, e.g.:
    - x, y, z: x-, y- or z-coordinate in cartesian coordinates
    - a, b, c: direct coordinates, i.e. fractions of the lattice vectors, for the first, second or third lattice vector
    - element: element of the atom
    - dof: degree of freedom, i.e. is the atom allowed to move in the said direction
    - velocity: velocity of the atom
    - atoms: 0-indexed list of atoms

    These attributes can be used with additional comparison operators which are appended to the argument name with "__".
    Possible operators:
    - eq: equal: the argument has to equal the exact value. This is the default operator in case no operator is specified
    - lt: lower than: the argument hast to be lower than the given value
    - le: lower equal: the argument hast to be lower than or equal the given value
    - gt: greater than: the argument hast to be greater than the given value
    - ge: greater equal: the argument hast to be greater than or equal the given value

    Examples:
    >>> @selectable()
    >>> def f(crystal, vec1, vec2, atoms):
    ...     return crystal.atoms.position[atoms] * vec1 - vec2
    ...
    >>> f(cryst, np.array([1.0,0,0]), np.array([1.0,1.0,1.0]), element = ['H', 'C'], c__gt = 0.5, x__le = 5.0)

    This would apply the function f on the atoms in <crystal> which are hydrogen or carbon atoms, which lie in the second half of the
    crystal cell spanned by the third vector and which have an x-coordinate that is 5.0 Angstrom or below.
    The compiled index list is provdided through the <atoms> argument.

    The decorated function must provide the arguments <crystal> and <atoms>, while <atoms> must be the last argument.
    This behaviour can be altered by setting <crystalName> and <indexName> in decoration process:
    @selectable(crystalName='myCrystal', indexName='myIndices')
    The default is
    @selectable(crystalName='crystal', indexName='atoms')"""

    def __init__(self, crystalName='crystal', indexName='atoms'):
        self.crystalName = crystalName
        self.indexName = indexName

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(f)
            # separate parameters of f() and parameters for selecting atoms:
            keys = {}
            for key, val in kwargs.items():
                if key not in sig.parameters:
                    keys[key] = val
            # get difference of kwargs and keys
            kwargs = {k: v for k, v in kwargs.items() if k not in keys}

            # move all parameters from args to kwargs so we know which parameter is the crystal
            for n, val in enumerate(args):
                kwargs[list(sig.parameters.keys())[n]] = val

            # get the crystal
            try:
                crystal = kwargs[self.crystalName]
            except KeyError:
                raise ValueError(
                    '"Missing value for "%s" in call of %s.' % (self.crystalName, f))
            # now look for atom list
            atoms = kwargs.get(self.indexName, None)
            keys['atoms'] = atoms

            # collect the indices
            atoms = crystal.get_indices(**keys)

            kwargs[self.indexName] = atoms
            return f(**kwargs)
        return wrapper


class metaDirect(type):
    """meta class providing access to all attributes of <class numpy.ndarray>"""

    def __init__(cls, name, bases, dct):
        def make_proxy(name):
            def proxy(self, *args):
                return getattr(self._obj, name)
            return proxy

        type.__init__(cls, name, bases, dct)
        if cls.__wraps__:
            ignore = cls.__ignore__.split()
            for name in dir(cls.__wraps__):
                if name not in ignore and name not in dct:
                    setattr(cls, name, property(make_proxy(name)))


class DirectPosition(metaclass=metaDirect):
    """mapping direct coordinates interactively to cartesian coordinates"""

    __wraps__ = np.ndarray
    __ignore__ = "__class__ __mro__ __init__ __new__ __getattr__ __setattr__ __getattribute__ __sub__ __add__"

    @property
    def _obj(self):
        # calculate "fresh" direct coordinates
        return np.dot(
            np.linalg.inv(np.array(self._lattice, dtype=np.float64).T),
            self._coord.T
        ).T[self._indices[0], self._indices[1]]

    def __init__(self, coordinates, lattice, indices=None):
        self._coord = coordinates
        self._lattice = lattice
        self._indices = indices if indices is not None else np.indices(
            coordinates.shape)

        super().__init__()

    def __getitem__(self, key):
        return self.__class__(self._coord, self._lattice, [self._indices[0][key], self._indices[1][key]])

    def __setitem__(self, key, value):
        # get direct coordinates of the atoms
        direct = np.dot(
            np.linalg.inv(np.array(self._lattice, dtype=np.float64).T),
            self._coord.T
        ).T
        # set new values
        direct[self._indices[0][key], self._indices[1][key]] = value
        # write back to cartesian coordinates
        self._coord[:] = np.dot(self._lattice.T, np.array(
            direct, dtype=np.float128).T).T

    def __sub__(self, obj):
        return self._obj.__sub__(np.array(obj))

    def __add__(self, obj):
        return self._obj.__add__(np.array(obj))

    def as_np_array(self):
        return self._obj


class metaAtom(type):
    """meta class providing access to all attributes of <class numpy.ndarray>"""

    def __init__(cls, name, bases, dct):
        def make_proxy(name):
            def proxy(self, *args):
                return getattr(self._obj, name)
            return proxy

        type.__init__(cls, name, bases, dct)
        if cls.__wraps__:
            ignore = cls.__ignore__.split()
            for name in dir(cls.__wraps__):
                if name not in ignore and name not in dct:
                    setattr(cls, name, property(make_proxy(name)))


class Atoms(metaclass=metaAtom):
    """class for storing element, position, degree of freedom and velocity of atoms"""

    __wraps__ = np.ndarray
    __ignore__ = "__class__ __mro__ __init__ __new__ __getattr__ __setattr__ __getattribute__"

    def __init__(self, shape, parent):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instantiated")

        super().__init__()
        dtype = [('element', np.str, 2), ('position', np.float128, 3),
                 ('dof', np.bool, 3), ('velocity', np.float128, 3)]
        # avoid __setattr__ for attribute '_obj' due to infinite loop
        self.__dict__['_obj'] = np.ndarray(shape, dtype=dtype)
        self._parent = parent

    def __getattr__(self, attr):
        # lookup attribute in self._array
        try:
            return self._obj[attr]
        except ValueError:
            pass
        raise AttributeError(attr)

    def __setattr__(self, name, value):
        if name in self._obj.dtype.names:
            self._obj[name] = value
        else:
            super().__setattr__(name, value)

    def __getitem__(self, key):
        # allow keys of atom type, e.g., "H", "Au"
        if isinstance(key, str):
            mask = np.where(self.element == key)
            tmp = self._obj[mask]
        elif hasattr(key, '__iter__') and all(isinstance(x, str) for x in key):
            mask = np.array([], dtype=np.int)
            for el in key:
                mask = np.append(mask, np.where(self.element == el))
            tmp = self._obj[list_to_slice(np.unique(mask))]
        else:
            tmp = self._obj[key]

        view = Atoms(tmp.shape, self._parent)
        view._obj = tmp
        return view

    def __setitem__(self, key, value):
        if isinstance(key, str):
            mask = np.where(self.element == key)
        elif hasattr(key, '__iter__') and all(isinstance(x, str) for x in key):
            mask = np.array([], dtype=np.int)
            for el in key:
                mask = np.append(mask, np.where(self.element == el))
            mask = np.unique(mask)
        else:
            mask = key

        # differentiate between tuple and Atoms class
        if isinstance(value, Atoms):
            self._obj[mask] = value._obj
        else:
            self._obj[mask] = value

    def __deepcopy__(self, memo):
        cp = Atoms(0, self._parent)
        cp._obj = copy.deepcopy(self._obj, memo=memo)
        return cp

    @property
    def direct(self):
        """returns direct coordinates of all atoms"""
        return DirectPosition(self.position, self._parent.lattice)

    @direct.setter
    def direct(self, value):
        """sets the direct coordinates of all atoms"""
        self.position = np.dot(self._parent.lattice.T,
                               np.array(value, dtype=np.float128).T).T

    @property
    def x(self):
        """returns the x position of all atoms in cartesian coordinates"""
        return self.position[..., 0]

    @x.setter
    def x(self, value):
        """accepts atom x-positions in cartesian coordinates"""
        self.position[:, 0] = value

    @property
    def y(self):
        """returns the y position of all atoms in cartesian coordinates"""
        return self.position[..., 1]

    @y.setter
    def y(self, value):
        """accepts atom y-positions in cartesian coordinates"""
        self.position[:, 1] = value

    @property
    def z(self):
        """returns the z position of all atoms in cartesian coordinates"""
        return self.position[..., 2]

    @z.setter
    def z(self, value):
        """accepts atom z-positions in cartesian coordinates"""
        self.position[:, 2] = value

    @property
    def a(self):
        """returns the first coordinate of all atoms in direct coordinates"""
        return self.direct[..., 0]

    @a.setter
    def a(self, value):
        """accepts first position parameter in direct coordinates"""
        self.direct[:, 0] = value

    @property
    def b(self):
        """returns the second coordinate of all atoms in direct coordinates"""
        return self.direct[..., 1]

    @b.setter
    def b(self, value):
        """accepts second position parameter in direct coordinates"""
        self.direct[:, 1] = value

    @property
    def c(self):
        """returns third coordinate of all atoms in direct coordinates"""
        return self.direct[..., 2]

    @c.setter
    def c(self, value):
        """accepts third position parameter in direct coordinates"""
        self.direct[:, 2] = value


class Crystal:
    # class containing all the information of a PROCAR file
    def __init__(self, filename=None, title='', scaling=1.0, lattice=np.zeros(9, dtype=np.float128).reshape(3, 3),
                 atoms=None, selective_dynamics=False, cartesian=False):
        """If filename is given all the data is read from the file. Otherwise the other remaining parameters are taken"""
        if filename:
            self.read(filename)
        else:
            self.title = title

            self.scaling = scaling
            self.lattice = scaling * np.array(lattice)
            if lattice.shape != (3, 3):
                print('lattice must have shape of (3, 3)')
                raise ValueError

            if isinstance(atoms, Atoms):
                self.atoms = atoms
                self.atoms._parent = self
            else:
                self.atoms = Atoms(0, self)

            self.selective_dynamics = selective_dynamics
            self.cartesian = cartesian

    @property
    def elements(self):
        indices = np.unique(self.atoms.element, return_index=True)[1]
        return [self.atoms.element[index] for index in sorted(indices)]

    @property
    def number_atoms(self):
        result = np.column_stack(
            np.unique(self.atoms.element, return_counts=True))
        return dict([(x[0], int(x[1])) for x in result])

    @property
    def number_all_atoms(self):
        return self.atoms.shape[0]

    @property
    def reciprocal_lattice(self):
        return 2 * np.pi * np.transpose(np.linalg.inv(np.array(self.lattice, dtype=np.float64)))

    def read(self, filename):
        """read the POSCAR file <filename>"""
        self._filename = filename
        try:
            infile = open(self._filename, 'r')
        except FileNotFoundError:
            print('File ' + self._filename + ' not found!')
            raise

        # read title
        self.title = infile.readline()

        # read scaling factor
        self.scaling = float(infile.readline().strip())

        # read lattice vectors
        self.lattice = np.empty(9, dtype=np.float128).reshape(3, 3)
        for i in range(3):
            self.lattice[i] = infile.readline().strip().split()
        self.lattice *= self.scaling

        # read elements
        elements = self._read_elements(infile)

        # read number of atoms
        self._number_of_atoms = [
            int(x) for x in infile.readline().strip().split()]
        if len(elements) != len(self._number_of_atoms):
            raise RuntimeError(
                "Number of elements and number of rows of atom numbers do not match")

        # check for selective dynamics
        pos = infile.tell()
        if infile.readline()[0] not in ('S', 's'):
            infile.seek(pos)
            self.selective_dynamics = False
        else:
            self.selective_dynamics = True

        # check for coordinate system
        if infile.readline()[0] not in ('C', 'c', 'K', 'k'):
            self.cartesian = False
        else:
            self.cartesian = True

        # read atom coordinates
        self.atoms = Atoms(sum(self._number_of_atoms), self)
        for (i, el) in enumerate(elements):
            start = sum(self._number_of_atoms[:i])
            for n in range(start, start + self._number_of_atoms[i]):
                line = infile.readline().strip().split()
                if self.selective_dynamics:
                    try:
                        self.atoms[n] = (elements[i],
                                         line[:3],
                                         [True if x ==
                                             'T' else False for x in line[3:6]],
                                         (0., 0., 0.))
                    except ValueError:  # check if information for dof is missing
                        self.atoms[n] = (elements[i],
                                         line[:3],
                                         (True, True, True),
                                         (0., 0., 0.))
                        # if we came this far without raising a new exception, we are probably missing the dofs
                        self.selective_dynamics = False
                else:
                    self.atoms[n] = (elements[i],
                                     line[:3],
                                     (True, True, True),
                                     (0., 0., 0.))
        # internally, coordinates are always stored in cartesian form
        if not self.cartesian:
            self.atoms.position = np.dot(
                self.lattice.T, self.atoms.position.T).T

        # read velocities
        infile.readline()  # skip empty line
        for (i, line) in enumerate(infile):
            if i >= self.atoms.shape[0]:
                break
            self.atoms.velocity[i] = tuple(line.strip().split()[:3])

        infile.close()

    def _read_elements(self, infile):
        """check if line with elements is given in POSCAR and if not, look for a POTCAR file"""
        pos = infile.tell()
        elements = infile.readline().strip().split()
        if any(x.isdigit() for x in elements):
            # no line with elements in POSCAR
            infile.seek(pos)
            try:
                elements = self._read_elements_from_potcar(
                    (dirname(infile.name) or '.') + '/POTCAR')
            except FileNotFoundError:
                raise FileNotFoundError(
                    'Could not find the elements of atoms. Provide them in the POSCAR or in a POTCAR file')

        # check if elements exist
        for element in elements:
            if element not in atomic_number:
                raise RuntimeError(
                    'Undefined element found in POSCAR file: {}'.format(element))
        return elements

    def _read_elements_from_potcar(self, filename):
        """reads a POTCAR and looks for the element"""
        potcar = None
        potcar = open(filename, "r")
        print("    reading POTCAR...")
        res = re.findall("VRHFIN =([A-Z][a-z]{0,1}):", potcar.read())
        print("      found atom types: ", ", ".join(res) + "\n")

        if potcar is not None:
            potcar.close()

        return res

    def write(self, filename=None):
        """write data to POSCAR file
        if no filename is given, the original file is overwritten"""
        filename = filename or self._filename
        with open(filename, 'w') as outfile:
            # title
            print(self.title.strip() or 'Unknown', file=outfile)
            # scaling factor
            print("{:19.14f}".format(self.scaling), file=outfile)
            # lattice vectors
            print("   {:20.16f}  {:20.16f}  {:20.16f}\n   {:20.16f}  {:20.16f}  {:20.16f}\n   {:20.16f}  {:20.16f}  {:20.16f}".format(
                *(self.lattice / self.scaling).flatten()), file=outfile)
            # elements and number of atoms
            strElement = ""
            strNumber = ""
            for i in range(len(self.elements)):
                strElement += " {:>4}"
                strNumber += " {:5d}"
            print(strElement.format(*self.elements), file=outfile)
            print(strNumber.format(*[self.number_atoms[key]
                                     for key in self.elements]), file=outfile)
            # selective dynamics
            if self.selective_dynamics:
                print("Selective dynamics", file=outfile)
            # coordinate system
            print("{:s}".format(
                "Cartesian" if self.cartesian else "Direct"), file=outfile)
            # atomic coordinates
            if self.cartesian:
                coordinates = self.atoms.position
            else:
                coordinates = self.atoms.direct
            if self.selective_dynamics:
                for i in range(len(self.atoms)):
                    print(" {:19.16f} {:19.16f} {:19.16f} {:>3s} {:>3s} {:>3s}".format(*coordinates[i], *['T' if x else 'F' for x in self.atoms[i].dof]),
                          file=outfile)
            else:
                for coordinate in coordinates:
                    print(" {:19.16f} {:19.16f} {:19.16f}".format(
                        *coordinate), file=outfile)
            # velocities
            if np.any(self.atoms.velocity != 0.0):
                print('', file=outfile)
                for i in range(len(self.atoms)):
                    print("  {:.8E}  {:.8E}  {:.8E}".format(
                        *self.atoms[i].velocity), file=outfile)

    def write_xsf(self, filename=None):
        """
        Write data to an .xsf file.
        If no file name is given, the original file name with an .xsf ending is used.
        """
        filename = filename or (self._filename + '.xsf')
        with open(filename, 'w') as outfile:
            outfile.write('# ' + self.title)
            outfile.write('CRYSTAL\n')
            # write lattice vectors
            outfile.write('PRIMVEC\n')
            for i in range(3):
                outfile.write(
                    '   {:9.7f}   {:9.7f}   {:9.7f}\n'.format(*self.lattice[i]))
            # write position
            outfile.write('PRIMCOORD\n')
            outfile.write('   {:d} 1\n'.format(self.number_all_atoms))
            start = 0
            # loop over different elements
            for j, el in enumerate(self.elements):
                # translate element name to atomic number
                atnum = atomic_number[el]
                # loop over atom position of element el
                for n in range(start, start + self.number_atoms[el]):
                    outfile.write('{:<3d} {:> 11.7f}   {:> 11.7f}   {:> 11.7f}\n'.format(
                        atnum, *self.atoms.position[n]))
                start += self.number_atoms[el]

    def get_indices(self, silent=False, **kwargs):
        """select specific atoms, based on the attributes found in <class Atoms>.
If <silent> is True, no exception is raised when one of the parameters is no attribute of <class Atoms>."""
        operators = {'eq': '__eq__', 'lt': '__lt__', 'le': '__le__',
                     'gt': '__gt__', 'ge': '__ge__'}  # operations permitted
        indices = np.arange(self.number_all_atoms)  # selected atoms

        # treat keyword "atoms" sperately
        if 'atoms' in kwargs:
            value = kwargs['atoms']
            if value is not None:
                # atoms can be supplied as strings (e.g. "1,2,5-10,14") or as integers or lists of integers
                if isinstance(value, str):
                    cleaned_val = value.replace(" ", "")
                    expansions = re.findall(r"\d+-\d+", cleaned_val)
                    try:
                        for i in expansions:
                            num1, num2 = [int(x) for x in i.split('-')]
                            sgn = np.sign(num2 - num1)
                            cleaned_val = re.sub(r"{:d}-{:d}".format(num1, num2),
                                                 ",".join([str(x) for x in range(num1, num2 + sgn, sgn)]), cleaned_val)
                        new_indices = [int(x) for x in cleaned_val.split(',')]
                    except:
                        print('"Error while expanding expression "' + str(i) + '"')
                        raise
                else:
                    try:
                        new_indices = [int(x) for x in value]
                    except TypeError:
                        new_indices = [int(value), ]
                indices = np.intersect1d(indices, new_indices)
            del kwargs['atoms']

        # now treat the remainder
        for key, value in kwargs.items():
            # separate operator from key
            if len(key.split('__')) > 1:
                key, op = key.split('__')
                if op not in operators:
                    raise KeyError("unrecognized operator: {:s}".format(op))
            else:
                op = 'eq'
            op = operators[op]
            # check for legit key
            if not hasattr(self.atoms, key) or callable(getattr(self.atoms, key)):
                if silent:
                    continue
                else:
                    raise KeyError("unrecognized key: {:s}".format(key))
            attribute = getattr(self.atoms, key)

            # make sure value is iterable
            if isinstance(value, str) or not hasattr(value, '__iter__'):
                value = [value, ]
            # now loop through the values
            new_indices = np.array([], dtype=np.int32)
            for val in value:
                new_indices = np.union1d(new_indices, np.where(
                    getattr(attribute, op)(val))[0])
            indices = np.intersect1d(indices, new_indices)
        return indices

    # define an alias of get_indices for backward compatibility
    argatoms = get_indices

    def get_atoms(self, **kwargs):
        args = self.get_indices(**kwargs)
        return self.atoms[args]

    @selectable('self')
    def copy(self, atoms=None):
        """Get a copy of this instance. If atoms specified, only these are copied."""
        new = copy.deepcopy(self)
        new.atoms._obj = new.atoms._obj[atoms]
        new.atoms._parent = new
        return new

    def arg_sort(self, sortVectors=np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]), gap=0.5, elementBlock=True):
        """sorts the atoms with respect to <sort_vectors> and returns the indices for this sorting.
        If <elementBlock> True, atoms are first sorted by element an then by <sort_vectors>"""

        # define a function sorting all atoms within precision
        def sort_precision(atoms, direction):
            fields = ('c', 'b', 'a')
            # sort in direct coordinates of sorting vectors
            direct = np.rec.fromarrays(atoms.direct.T, dtype=[
                ('a', np.float128), ('b', np.float128), ('c', np.float128)])
            args = np.argsort(direct, order=fields[2 - direction:])
            sortedAtoms = atoms[args]

            if direction > 0:
                # slice atoms in groups separated by <gap> in direction <direction>
                groups = []
                indices = np.squeeze(np.argwhere(
                    (sortedAtoms.direct[1:] - sortedAtoms.direct[:-1])[:, direction] > gap[direction - 1]),
                    axis=1)

                start = 0
                for index in indices:
                    groups.append(slice(start, index + 1))
                    start = index + 1
                groups.append(slice(start, len(atoms)))

                # now for every group, resort and split according to the remaining dimensions
                for group in groups:
                    subArgs = sort_precision(sortedAtoms[group], direction - 1)
                    args[group] = args[group][subArgs]
            return args

        # norm the sorting vectors
        normed = np.divide(
            np.array(sortVectors, dtype=np.float64),
            np.broadcast_to(np.linalg.norm(sortVectors, axis=1), (3, 3)).T
        )

        # check for linear independent vectors
        if not np.linalg.det(normed):
            raise ValueError('Sorting vectors need to be linear independent')

        # make the gap parameter a duple
        if not hasattr(gap, '__iter__'):
            gap = (gap, gap)

        # swap lattice vectors with sorting vectors and work in direct coordinates
        old_lattice = self.lattice
        self.lattice = normed

        if elementBlock:
            sortingArgs = np.arange(self.number_all_atoms, dtype=np.int32)[
                self.arg_sort_elements()]
            for el in self.elements:
                tmp = self.copy()
                tmp.atoms = self.atoms[sortingArgs]
                elArgs = tmp.get_indices(element=el)
                subArgs = sort_precision(self.atoms[sortingArgs][elArgs], 2)
                sortingArgs[elArgs] = sortingArgs[elArgs][subArgs]
        else:
            sortingArgs = sort_precision(self.atoms, 2)

        # restore lattice vectors
        self.lattice = old_lattice

        return sortingArgs

    def sort(self, sortVectors=np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]), gap=0.5, elementBlock=True):
        """sorts the atoms with respect to <sort_vectors>
        If <elementBlock> True, atoms are first sorted by element an then by <sort_vectors>"""
        args = self.arg_sort(sortVectors, gap, elementBlock)
        self.atoms = self.atoms[args]

    def arg_sort_elements(self, preserveOrder=True):
        """
        Returns indices for sorting the atoms so that atoms of one element are aranged in one block.
        If <preserveOrder> is True, elements are ordered as they first appear in self.atoms.
        Otherwise, elements are sorted in alphabetically."""
        if not preserveOrder:
            return np.argsort(
                self.atoms._obj, kind='mergesort', order='element')
        else:
            args = np.zeros(0, np.int32)
            for el in self.elements:
                args = np.append(args, np.argwhere(self.atoms.element == el))
            return args

    def sort_elements(self, preserveOrder=True):
        """
        Sorts the atoms so that atoms of one element are aranged in one block.
        If <preserveOrder> is True, elements are ordered as they first appear in self.atoms.
        Otherwise, elements are sorted in alphabetically."""
        args = self.arg_sort_elements(preserveOrder)
        self.atoms = self.atoms[args]

    @selectable('self')
    def reduce(self, atoms=None):
        """moves atoms outside the unit cell back inside the unit cell"""
        self.atoms.direct[atoms, :] -= np.floor(
            self.atoms.direct._obj[atoms])

    @selectable('self')
    def cut(self, atoms=None):
        ops = [{'a__gt': 1.0}, {'a__lt': 0.0}, {'b__gt': 1.0},
               {'b__lt': 0.0}, {'c__gt': 1.0}, {'c__lt': 0.0}]
        for op in ops:
            idx = self.get_indices(atoms=atoms, **op)
            self.atoms._obj = np.delete(self.atoms._obj, idx)

    def distance(self, atom_index1, atom_index2, periodic=True):
        """calculate the distance between two atoms"""
        dif = self.atoms.direct[atom_index2] - self.atoms.direct[atom_index1]
        if periodic:
            dif -= np.around(dif)

        return np.linalg.norm(np.dot(self.lattice.T, np.array(dif, dtype=np.float128).T).T)

    @selectable('self')
    def closest_to_points(self, points, direct=False, numNearest=None, distance=False, atoms=None):
        """Return the indices of <numNearest> atoms ordered by their distance
        to each of the N points in <points> with shape (N, 3).
        If <direct> is True, the coordinates in <points> are expected to be direct.
        """
        if not direct:
            # convert points from cartesian to direct coordinates
            dpoints = cartesian_to_direct(
                points, self.lattice.astype(np.float64))
        else:
            dpoints = points

        if dpoints.ndim == 2:
            dpoints = dpoints.reshape(-1, 1, 3)
        else:
            dpoints = dpoints.reshape(1, 3)

        # transform shape of direct coordinates
        direct = self.atoms.direct.as_np_array().reshape(1, -1, 3)

        # calculate connecting vector in direct coordinates
        con_vec = direct - dpoints
        # consider periodic boundary conditions
        con_vec -= np.around(con_vec)

        # transform to cartesian coordinates and calculate the distance
        dist = np.linalg.norm(direct_to_cartesian(
            con_vec, self.lattice.astype(np.float64)), axis=-1)

        # sort according to the distance
        args = np.argsort(dist)

        if distance:
            return args[:, :numNearest], np.sort(dist)[:, :numNearest]
        else:
            return args[:, :numNearest]

    @selectable('self', 'atoms')
    def nearest_neighbor(self, atoms=None, candidates=None, numNeighbors=None, distance=False):
        """Returns the indices of the <numNeighbors> nearest neighbors of <from_atoms>
        among the candidates of <candidates>.
        <from_atoms> can be a list of indices or a dict with keyword specifying the atoms.
        If numNeighbors=0, the ordered indices of all atoms are returned.
        Set distance=True to get a tuple of ordered indices and orderes distances.
        """
        if type(candidates) is dict:
            candidates = self.get_indices(**candidates)
        elif candidates is None:
            candidates = np.arange(self.number_all_atoms)

        try:
            numCandidates = len(candidates)
        except TypeError:
            raise TypeError(
                '<candidates> must be a list of indices or a dict with atom specifying keywords')

        grid = np.repeat(
            self.atoms.direct[atoms].as_np_array(),
            numCandidates,
            axis=0)
        grid = grid.reshape(len(atoms), numCandidates, 3)

        dif = grid - self.atoms[candidates].direct.as_np_array()
        dif -= np.around(dif)
        dist = np.linalg.norm(np.dot(self.lattice.T, np.swapaxes(
            np.array(dif, dtype=np.float128), 1, 2)), axis=0)

        # replace distance of 0 with the length of the smallest lattice vector
        dist[dist == 0] = np.min(np.linalg.norm(self.lattice, axis=1))
        ind = np.argsort(dist, axis=1)[:, :(
            numNeighbors) if numNeighbors else None]
        if distance:
            return (candidates[ind], np.take_along_axis(dist, ind, axis=1))
        else:
            return candidates[ind]

    def scale(self, scaling_factor):
        """scales the whole cell by the factor <scaling_factor>"""
        # make a copy of the atoms in direct representation
        direct = self.atoms.direct._obj
        # now scale lattice and restore old direct positions
        self.lattice *= scaling_factor
        self.atoms.direct = direct

    def visualize(self, prog='vesta'):
        """visualize the structure with VESTA"""

        def background():
            print('running in background')
            tmpFilename = 'tmp_' + str(hash(self))[:16] + '.xsf'
            self.write_xsf(tmpFilename)
            if prog == 'vesta':
                os.system(
                    '/Applications/VESTA/VESTA.app/Contents/MacOS/VESTA ' + tmpFilename)
            elif prog == 'xcrys':
                os.system('xcrysden --xsf ' + tmpFilename)
            else:
                print('Program "' + prog + '" is not supported')
            os.remove(tmpFilename)

        thread = Thread(target=background, args=())
        thread.daemon = True
        thread.start()

    def get_periodic_groups_of_atom(self, atom, pool=None, direction=None, precision=0.1):
        """
        Analyzes the periodicity of <atom> and returns all atoms among <pool> that are periodic copys along <direction>.
        If <direction> is None, the 3-tuple of the atoms along all 3 directions is returned.
        If <pool> is None, all atoms are analyzed.
        <precision> is the maximal lateral distance in Angstrom an atom is allowed to deviate
        """

        # which atoms should be checked?
        pool = np.arange(self.number_all_atoms) if pool is None else np.array(
            [pool]).flatten()
        # no need to include <atom> in <pool>
        pool = np.delete(pool, np.argwhere(pool == atom))

        # set up an array containing the directions and an array storing the groups of atoms
        direction = np.arange(3) if direction is None else np.array(
            [direction]).flatten()
        groups = []

        # treat each direction separately
        for i in direction:
            # calculate distance between <atom> and atoms in <pool> perpendicular to the <i>th lattice vector
            distance = self.atoms.direct[pool] - self.atoms.direct[atom]
            # no need in considering distances along direction <i>
            distance[:, i] = 0

            # take periodic boundaries into consideration
            distance -= np.around(distance)

            # calculate the distance in cartesian coordinates
            cartDist = np.linalg.norm(
                direct_to_cartesian(distance, self.lattice), axis=1)

            # now check which atoms "sit along the lattice vector"
            args = np.argwhere(cartDist < precision).flatten()
            group = np.append(pool[args], atom)

            # we need to check if the grouped atoms really are periodic, i.e. if they are equidistant
            # precision in direct coordinates
            precDirect = precision / np.linalg.norm(self.lattice[i])

            direct1D = self.atoms.direct[group, i].copy()
            args = np.argsort(direct1D)
            direct1D = direct1D[args]

            # possible periodicities are all divisors of the number of atoms of the group
            maxPeriodicity = len(group)
            for p in divisorGenerator(maxPeriodicity):
                fracLat = 1.0 / p
                regrouped = np.array(direct1D).reshape(p, -1).T
                regrouped = np.append(
                    regrouped, regrouped[:, 0].reshape(-1, 1) + 1.0, axis=1)  # make it periodic
                distance1D = np.diff(regrouped, axis=1)

                if np.all(np.isclose(distance1D, fracLat, rtol=0, atol=precDirect)):
                    # we find the periodicity. Now extract the correct group of indices
                    indices = group[args].reshape(p, -1).T
                    groups.append(
                        indices[np.argwhere(group[args].reshape(p, -1).T == atom).flatten()[0]])
                    break

        return groups

    def get_periodic_groups(self, pool=None, direction=None, merge=True, precision=0.4):
        """
        analyzes the periodicity of each atom in <pool> and splits them into groups that are periodic copys along <direction>.
        If <direction> is None, the 3-tuple of the atoms along all 3 directions is returned.
        If <pool> is None, all atoms are analyzed.
        Set <merge> to False to prohibit the merging of groups in different directions.
        <precision> is the maximal lateral distance in Angstrom an atom is allowed to deviate
        """
        # ################################################ #
        # ############### Helper functions ############### #
        # ################################################ #

        # function for indexing the groups of each atom and returning the periodicities
        def index_groups(groups):
            period = np.zeros(shape=(self.number_all_atoms,
                                     len(groups)), dtype=np.int)
            index = np.zeros(shape=(self.number_all_atoms,
                                    len(groups)), dtype=np.int)
            index[:, :] = -1
            for i in range(len(groups)):
                for n, group in enumerate(groups[i]):
                    lg = len(group)
                    for atom in group:
                        period[atom, i] = lg
                        index[atom, i] = n
            return index, period

        # function for finding duplicate atoms in groups of a single direction and resolving these conflicts
        def find_duplicates(supergroup):
            for groups in supergroup:
                if not np.array_equal(np.sort(np.hstack(groups)), np.sort(pool)):
                    # find out which indices occur multiple times
                    idx, cts = np.unique(np.hstack(groups), return_counts=True)
                    idx = list(idx[np.argwhere(cts > 1).flatten()])

                    # now deal with each atom
                    while(idx):
                        atom = idx.pop()

                        # find groups with these indices and try to merge them
                        conflictGroups = []
                        for group in groups:
                            if atom in group:
                                conflictGroups.append(group)

                        # is there a superset of the remaining sets?
                        # (N)umber of elements of the (L)argest (G)roup
                        nlg = max([len(x) for x in conflictGroups])
                        # (N)umber of (U)nique (E)lements
                        nue = len(np.unique(np.hstack(conflictGroups)))
                        if nlg == nue:
                            # superset exists. Remove the remaining groups
                            for group in groups:
                                if atom in group and len(group) != nlg:
                                    groups.remove(group)
                        else:
                            print(
                                'Groups are ambiguous. Please consider changing <precision>.')
                            raise ValueError

        # checks that all atoms within one periodic group in one direction have the same periodicity in the other directions
        def check_periodicity_in_3D(groups):
            # first index all groups for each direction
            index, period = index_groups(groups)

            for i in range(len(groups)):
                deletedGroups = []  # groups that have been further split and need to be deleted at the end
                for group in groups[i]:
                    # have atoms have different periodicities?
                    groupPeriods = period[list(group), :]
                    p, idx, cts = np.unique(
                        groupPeriods, return_index=True, return_counts=True, axis=0)

                    if cts[0] != len(group):
                        # can they be further split?
                        try:
                            # subsets must be of equal size with constant stride within the superset
                            nSplits = len(group) // min(cts)
                            reshapedPeriods = groupPeriods.reshape(
                                -1, nSplits, len(groups))
                            if len(np.unique(reshapedPeriods, axis=0)):
                                for j in range(nSplits):
                                    groups[i].append(group[j::nSplits])
                            else:
                                raise ValueError
                        except ValueError:
                            # if we are here, the group has no inherit periodicity
                            for j in group:
                                groups[i].append((group[j], ))
                        # delete group later to prevent index shifting in <index>
                        deletedGroups.append(group)

                # now remove the splitted groups from the list
                for group in deletedGroups:
                    groups[i].remove(group)

        # merge groups for different directions
        def merge_groups(groups):
            # resort and reindex groups
            for group in groups:
                group.sort(key=lambda x: x[0])
            index, period = index_groups(groups)

            # work through all directions backwards
            while(len(groups) > 1):
                merged = []
                dim1 = len(groups) - 2
                dim2 = dim1 + 1

                # loop through all groups of direction <dim1> and merge them with groups of direction <dim2>
                while(groups[dim1]):
                    # pick one group of direction 1
                    firstGroup1 = groups[dim1][0]
                    # get all groups in direction 2 which intersect with <firstGroup1>
                    groups2 = [groups[dim2][x]
                               for x in index[list(firstGroup1), dim2]]
                    # of the above groups pick one in direction 2
                    firstGroup2 = groups2[0]
                    # get all groups in direction 2 which intersect with <firstGroup2>
                    groups1 = [groups[dim1][x]
                               for x in index[list(firstGroup2), dim1]]

                    # the union of groups1 and groups2 must be equal
                    if not np.array_equal(
                            np.sort(np.hstack(groups1)),
                            np.sort(np.hstack(groups2))
                    ):
                        raise ValueError

                    merged.append(tuple(np.sort(np.hstack(groups1))))

                    # now delete all groups that have been merged
                    for group in groups1:
                        groups[dim1].remove(group)
                    for group in groups2:
                        groups[dim2].remove(group)

                    index, _ = index_groups(groups)

                # all groups of direction <dim1> and <dim2> have been processed
                del groups[dim2]
                groups[dim1] = merged
                index, _ = index_groups(groups)

        # ######################################################### #
        # ############### Main part of the function ############### #
        # ######################################################### #

        pool = np.arange(self.number_all_atoms) if pool is None else np.array(
            [pool]).flatten()
        direction = np.arange(3) if direction is None else np.array(
            [direction]).flatten()

        # get groups for each atom:
        groups = [list({tuple(self.get_periodic_groups_of_atom(
            atom, pool, d, precision)[0]) for atom in pool}) for d in direction]

        # sort the groups per index for each direction and make sure every atom is exactly included once
        for group in groups:
            group.sort(key=lambda x: x[0])
        find_duplicates(groups)

        # merge groups of different directions into one set of groups
        if merge:
            check_periodicity_in_3D(groups)
            merge_groups(groups)

        return groups[0]


# for consistency with older versions. Deprecated!
class poscar(Crystal):
    def __init__(self):
        print('Use of class <poscar> is deprecated and will be removed in the future. Use class <Crystal> instead.')
        self.super().__init__()
    pass


@selectable()
def translate(crystal, vector, reduce=True, atoms=None):
    """
    Translate all atoms in <crystal> by the <vector> in cartesian coordinates.
    If <reduce> is set to False, the atoms will not be shifted back into the unit cell.
    """
    new = crystal.copy()
    new.atoms.position[atoms] = crystal.atoms.position[atoms] + vector

    if reduce:
        new.reduce()

    return new


@selectable('crystal2')
def append(crystal1, crystal2, atoms=None, sort=True):
    """Add the atoms of <crystal2>, selected by atoms, to <crystal1>"""
    new = crystal1.copy()
    new.atoms._obj = np.append(crystal1.atoms._obj, crystal2.atoms._obj[atoms])
    if sort:
        new.sort_elements()
    return new


@selectable()
def multiply(crystal, multiplicator, shift=[1.0, 1.0, 1.0], atoms=None, sort=True):
    """Multiplies <crystal> by <multiplicator> ([ma, mb, mc] or m) along the lattice vectors.
Set <shift> to move the replicated part of the crystal by more/less than one lattice vector.
Example:
    Multiply three times in direction of the first lattice vector and two times in direction of the second lattice vector
    >>> multiply(crystal, [3, 2, 1])

    Multiply 2 times in all three directions with a shift of 1.5 for the second lattice vector
    >>> multiply(crystal, 2, shift=[1.0, 1.5, 1.0])"""
    # check and set multiplicator
    if not np.all(np.array(multiplicator) > 0):
        raise ValueError("<multiplicator> must be set to a value greater 0")
    if not hasattr(multiplicator, '__iter__'):
        multiplicator = [multiplicator, multiplicator, multiplicator]
    # check and set shifted
    if not np.all(np.array(shift) > 0):
        raise ValueError("<shift> must be set to a value greater 0")
    if not hasattr(shift, '__iter__'):
        shift = [shift, shift, shift]

    new = crystal.copy()  # original cell where atoms are added
    # cell containing only atoms that are shifted
    mul = crystal.copy(atoms=atoms)

    # for each direction, duplicate the crystal, shift and append to original cell
    for i in range(3):  # loop over the three lattice vectors
        local = mul.copy()
        # loop over the number of repetitions
        for j in range(1, multiplicator[i]):
            local.atoms.direct[:, i] += shift[i]
            mul = append(mul, local, sort=False)
    # remove the "original" unshifted atoms from <mul> and add to <new>
    mul = delete(mul, list(range(len(atoms))))
    new = append(new, mul, sort=False)

    # sort the atoms and adjust the lattice
    if sort:
        new.sort_elements()
    new.lattice = np.multiply(new.lattice.T,
                              np.ceil([x.max() for x in new.atoms.direct.T])).T
    return new


@selectable()
def delete(crystal, atoms):
    """Deletes <atoms> from the crystal"""
    new = crystal.copy()
    new.atoms._obj = np.delete(new.atoms._obj, atoms)
    return new
