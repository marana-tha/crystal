#!/usr/bin/env python3

# read the phonon eigenvalues and eigenvectors from the OUTCAR
# ----------------------------
# written by Christian Braun
#    christian.braun@upb.de
# ----------------------------

from .parameters import atomic_mass

import numpy as np
import re


class OutcarParseError(Exception):
    pass


class Phonon():
    # class containing the phonon eigenvectors and eigenvalues
    def __init__(self, filename):
        self._normalized_by_mass = False
        self.read(filename)

    def read(self, filename):
        """
        read phonon eigenvalues and eigenvectors from outcar
        """
        try:
            outcar = open(filename, 'r+')
        except FileNotFoundError:
            print('File ' + filename + ' not found!')
            raise

        # find out how many ions we have
        nions = 0
        for line in outcar:
            if 'NIONS' in line:
                match = re.search(r'NIONS\s*=\s*(\d+)', line)
                nions = int(match.groups()[0])
                break
        else:
            raise OutcarParseError('Missing tag "NIONS" in OUTCAR')

        # go to the section where the phonon data is
        for line in outcar:
            if 'Eigenvectors and eigenvalues of the dynamical matrix' in line:
                break
        else:
            raise OutcarParseError('No phonon data found.')
        for i in range(3):
            outcar.readline()

        # allocate the memory for the data
        dtype = np.dtype([('wavenumber', np.float64), ('displacement', np.float64, (nions, 3))])
        data = np.zeros(3 * nions, dtype)

        # read the data
        ndof = 0
        line = outcar.readline()
        while line:
            match = re.search(r'([-\d\.]+)\s*cm-1', line)
            if match is None:
                break
            data['wavenumber'][ndof] = float(match.groups()[0])

            outcar.readline()
            for i in range(nions):
                line = outcar.readline()
                data['displacement'][ndof, i, :] = [float(x) for x in line.strip().split()[3:]]
            ndof += 1
            outcar.readline()
            line = outcar.readline()

        self.data = data[:ndof]

    def normalize(self):
        """
        normalize the eigenvectors
        """
        disp = self.data['displacement']
        self.data['displacement'] = np.moveaxis(np.divide(
            np.moveaxis(disp, 0, -1),
            np.linalg.norm(disp, axis=(1, 2))
        ), -1, 0)

    def normalize_by_mass(self, crys):
        """
        divide each displacement by 1/sqrt(M) with M being the mass of the atom and then normalize the eigenvectors
        """
        try:
            # get the mass for each displacement vector
            masses = np.zeros(self.data['displacement'].shape[1])
            for i, el in enumerate(crys.atoms.element):
                masses[i] = atomic_mass[el]

            # divide displacement vector by its mass
            
            disp = np.moveaxis(
                np.divide(np.moveaxis(self.data['displacement'], 1, -1), masses),
                -1, 1)
        except ZeroDivisionError:
            print('Number ob atoms of phonon data and crys not compatible.')

        self.data['displacement'] = disp
        self.normalize()
        self._normalized_by_mass = True

    def compare(self, phonon):
        """
        compares the eigenvectors of self with the eigenvectors of <phonon> and returns the overlap
        """
        # reshape both displacement arrays and make them square

        dispA = self.data['displacement']
        dispA = dispA.reshape(dispA.shape[0], -1)
        dispA = np.squeeze(dispA[:, np.argwhere(np.any(dispA, axis=0))])
        dispB = phonon.data['displacement']
        dispB = dispB.reshape(dispB.shape[0], -1)
        dispB = np.squeeze(dispB[:, np.argwhere(np.any(dispB, axis=0))])

        overlap = np.dot(dispA,
                         np.linalg.inv(dispB))
        ovSq = np.square((overlap.T / np.linalg.norm(overlap, axis=1)).T)
        args = np.flip(np.argsort(ovSq, axis=1), axis=1)

        # print the data:
        for i in range(overlap.shape[0]):
            print(('Mode {:4d}: overlap=[' + '{:>6.2f}% ' * 5 + '], indices=[' + ' {:>4d}' * 5 + ' ]').format(i + 1, *(ovSq[i][args[i][:5]] * 100), *(args[i][:5] + 1)))

        return ovSq

    def project(self, atoms, print_data=True):
        """
        get the projection of each eingenvector onto <atoms>
        """
        # determine the projection of the multidimensional displacements onto the subroom spanned by <atoms>
        sel = self.data['displacement'][:, atoms, :]
        projection = np.linalg.norm(sel, axis=(1, 2)) / np.linalg.norm(self.data['displacement'], axis=(1, 2))

        if print_data:
            out = 'Mode {:4d}: {:>6.2f}%'
            for i, p in enumerate(projection):
                if p < 0.5:
                    clr = '\033[39m'
                elif p < 0.67:
                    clr = '\033[32m'
                elif p < 0.84:
                    clr = '\033[33m'
                else:
                    clr = '\033[31m'
                print(clr + out.format(i + 1, p * 100))

        return projection

    def check_periodicity(self, crystal, precision=0.1, print_data=True):
        """
        checks the periodicity of the eigenmodes by comparing them to the periodicity
        of the crystal.
        """
        def calc_overlap(disp):
            # calculate agreement in direction and length
            overlapMatrix = np.dot(disp, disp.T)
            diag = np.diag(overlapMatrix).copy()
            overlapMatrix[np.diag_indices(len(group))] = 0
            overlap = np.sum(overlapMatrix) / (len(group)**2 - len(group)) / np.average(diag)

            return overlap


        # get the periodic groups of the crystal for all atoms that are allowed to move
        atoms = np.argwhere(np.any(crystal.atoms.dof, axis=1)).flatten()
        groups = crystal.get_periodic_groups(pool=atoms, merge=True, precision=precision)

        # index the groups
        index = np.zeros(shape=(crystal.number_all_atoms), dtype=np.int)
        for i, group in enumerate(groups):
            for atom in group:
                index[atom] = i

        # we need normalized vectors
        self.normalize()

        eigenmodes = self.data['displacement']
        gammas = []

        for j, ev in enumerate(eigenmodes):
            gamma = 0.0  # value defining if the mode is a gamma mode
            percentage = 0.0  # percentage of the displacement vector that has been processed

            # find out which atoms have the highest contribution to the eigenvector <d>
            args = np.flip(np.argsort(np.linalg.norm(ev, axis=1)))
            processedAtoms = []
            n = 0

            while percentage < 0.80:  # 80% of the eigenvector must be taken into consideration
                while args[n] in processedAtoms:
                    n += 1

                group = groups[index[args[n]]]
                disp = ev[list(group)]

                # calculate overlap and total weight of the actual group
                overlap = calc_overlap(disp)
                weight = np.linalg.norm(disp)**2

                gamma += overlap * weight
                percentage += weight

                for atom in group:
                    processedAtoms.append(atom)

                n += 1

            # norm the gamma value by the percentage that was processed
            gammas.append(gamma / percentage)

        if print_data:
            print(groups)
            out = 'Mode {:4d}: {:>6.2f}%'
            for i, p in enumerate(gammas):
                if p < 0.5:
                    clr = '\033[39m'
                elif p < 0.67:
                    clr = '\033[32m'
                elif p < 0.84:
                    clr = '\033[33m'
                else:
                    clr = '\033[31m'
                print(clr + out.format(i + 1, p * 100))

        return gammas

    def check_periodicity_along_direction(self, crystal, direction, periodicity=1, precision=0.1, print_data=True):
        """
        checks the periodicity of the eigenmodes by comparing them to the periodicity
        of the crystal.
        """
        def calc_overlap(disp):
            # calculate agreement in direction and length
            j = disp.shape[0]
            if j > 1:
                overlapMatrix = np.dot(disp, disp.T)
                diag = np.diag(overlapMatrix).copy()
                overlapMatrix[np.diag_indices(j)] = 0
                overlap = np.sum(overlapMatrix) / (j**2 - j) / np.average(diag)
            else:
                overlap = 1.0

            return overlap

        # get the periodic groups of the crystal for all atoms that are allowed to move
        atoms = np.argwhere(np.any(crystal.atoms.dof, axis=1)).flatten()
        groups = crystal.get_periodic_groups(pool=atoms, direction=direction, merge=False, precision=precision)

        # index the groups
        index = np.zeros(shape=(crystal.number_all_atoms), dtype=np.int)
        for i, group in enumerate(groups):
            for atom in group:
                index[atom] = i

        # we need normalized vectors
        self.normalize()

        eigenmodes = self.data['displacement']
        gammas = []

        for j, ev in enumerate(eigenmodes):
#            if j == 606:
#                import pdb
#                pdb.set_trace()
            gamma = 0.0  # value defining if the mode is a gamma mode
            percentage = 0.0  # percentage of the displacement vector that has been processed

            # find out which atoms have the highest contribution to the eigenvector <d>
            args = np.flip(np.argsort(np.linalg.norm(ev, axis=1)))[:len(atoms)]
            processedAtoms = []
            n = 0

            while percentage < 0.80:  # 80% of the eigenvector must be taken into consideration
                try:
                    while args[n] in processedAtoms:
                        n += 1
                except IndexError:
                    if n != 204:
                        import pdb
                        pdb.set_trace()
                    break  # we already checked all atoms with matching preiodicity

                group = groups[index[args[n]]]
                # try to split the group in a smaller subgroup with matching periodicity
                if len(group) % periodicity:
                    # group does not have requested periodicity. Skip all atoms in this group
                    subgroup = list(group)
                    disp = ev[subgroup]

                    overlap = 0.0
                else:
                    stride = len(group) // periodicity
                    try:
                        subgroup = [[x for x in group[i::stride]] for i in range(stride) if args[n] in group[i::stride]][0]
                    except IndexError:
                        import pdb
                        pdb.set_trace()

                    disp = ev[subgroup]

                    # calculate overlap
                    overlap = calc_overlap(disp)

                # calculate total weight of the actual group
                weight = np.linalg.norm(disp)**2

                gamma += overlap * weight
                percentage += weight

                for atom in subgroup:
                    processedAtoms.append(atom)

                n += 1

            # norm the gamma value by the percentage that was processed
            gammas.append(gamma / percentage)

        if print_data:
            print(groups)
            out = 'Mode {:4d}: {:>6.2f}%'
            for i, p in enumerate(gammas):
                if p < 0.5:
                    clr = '\033[39m'
                elif p < 0.67:
                    clr = '\033[32m'
                elif p < 0.84:
                    clr = '\033[33m'
                else:
                    clr = '\033[31m'
                print(clr + out.format(i + 1, p * 100))

        return gammas
