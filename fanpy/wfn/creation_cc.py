"""Creation Coupled-Cluster wavefunction."""
import numpy as np
from itertools import combinations

from fanpy.wfn.base import BaseWavefunction
from fanpy.tools import slater


class CreationCC(BaseWavefunction):
    r"""Creation Coupled-Cluster wavefunction.

    The creation CC wavefunction is given by

    .. math::
        |\Psi \rangle = \exp(\sum_{ij}c_{ij}a^{\dagger}_i a^{\dagger}_j) | 0 \rangle
    
    where :math: | 0 \rangle is the vacuum state, :math: a^{\dagger}_i is the creation operator
    for the i-th spin orbital, and :math: c_{ij} are the parameters of the wavefunction.
    
    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals.
    memory : {float, int, str, None}
        Memory available for the wavefunction.
    dict_orbpair_ind : dict of 2-tuple of int: int
        Dictionary that maps orbital pairs to column indices.
    dict_ind_orbpair : dict of int: 2-tuple of int
        Dictionary that maps column indices to orbital pairs.
    params : np.ndarray
        Parameters of the wavefunction.
    permutations : list of list of 2-tuple of int
        Permutations of the orbital pairs.
    signs : list of int
        Signs of the permutations.
    
    Methods
    -------
    __init__(nelec, nspin, memory=None, orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_nelec(nelec)
        Assign the number of electrons.
    assign_params(params=None, add_noise=False)
        Assign the parameters of the wavefunction.
    assign_orbpairs(orbpairs=None)
        Assign the orbital pairs used to construct the wavefunction.
    get_col_ind(orbpair)
        Get the column index that corresponds to the given orbital pair.
    get_permutations()
        Get the permutations of the given indices.
    get_sign(indices)
        Get the sign of the permutation of the given indices.
    _olp(sd)
        Calculate overlap with Slater determinant.
    _olp_deriv(sd)
        Calculate the derivative of the overlap.
    get_overlap(sd, deriv=None) : {float, np.ndarray}
        Return the (derivative) overlap of the wavefunction with a Slater determinant.
    calculate_product(occ_indices, permutation, sign)
        Calculate the product of the parameters of the given permutation.
    """

    def __init__(self, nelec, nspin, quadruples=False, memory=None, clusters=None, params=None):
        """Initialize the wavefunction
        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            If number is provided, it is the number of bytes.
            If string is provided, it should end iwth either "mb" or "gb" to specify the units.
            Default does not limit memory usage (i.e. infinite).
        clusters : iterable of 2-tuple of ints or 4-tuple of ints, optional
            Indices of the orbital clusters that will be used to construct creation cc.
        params : np.ndarray
            Coefficients.
        """

        super().__init__(nelec, nspin, memory=memory)
        if type(quadruples) is not bool:
            raise TypeError("`quadruples` must be a boolean value.")
        self.quadruples = quadruples
        self.assign_clusters(clusters=clusters)
        self.assign_params(params=params)
        self.permutations, self.signs = self.get_permutations()
        
    def assign_nelec(self, nelec: int):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.
        NotImplementedError
            If number of electrons is odd.

        """

        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise NotImplementedError("Odd number of electrons is not supported.")

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the creation cc wfn.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the creation CC wavefunction. Default: None
        add_noise : bool
            Option to add noise to the given parameters. Default: False
        """

        elec_pairs = int(self.nelec / 2)
        if params is None:
            number_params = int(self.nspin * (self.nspin - 1) / 2)
            if self.quadruples:
                number_params += int(
                    self.nspin * (self.nspin - 1) * (self.nspin - 2) * (self.nspin - 3) / 24
                )
            params = np.zeros(number_params)
        orbpairs = []
        for i in range(elec_pairs):
            orbpairs.append((i, self.nspatial + i))
        orbpairs = np.array(orbpairs)
        orbpairs = orbpairs.flatten()
        orbpairs = np.sort(orbpairs)
        orbpairs = orbpairs.reshape((elec_pairs, 2))
        for pair in orbpairs:
            col_ind = self.get_col_ind(tuple(pair.tolist()))
            params[col_ind] = 1
        super().assign_params(params=params, add_noise=add_noise)

    def assign_clusters(self, clusters=None):
        """Assign the orbital clusters used to construct the creation CC wavefunction.

        Parameters
        ----------
        clusters : iterable of 2-tuple/list (4-tuple/list) of ints
            Indices of the orbital clusters that will be used to construct the wavefunction.
            Default is all possible orbital pairs and quadruples if enabled.

        Raises
        ------
        TypeError
            If `clusters` is not an iterable.
            If an orbital cluster is not given as a 2-tuple/list of integers.
        ValueError
            If an orbital cluster has the same integer.
            If an orbital cluster occurs more than once.

        Notes
        -----
        Must have `nspin` defined for the default option.

        """
        if clusters is None:
            clusters = list((i, j) for i in range(self.nspin) for j in range(i + 1, self.nspin))
            if self.quadruples:
                quad_clusters = list((i, j, k, l) for i in range(self.nspin) for j in range(i + 1, self.nspin) for k in range(j + 1, self.nspin) for l in range(k + 1, self.nspin))
                clusters.extend(quad_clusters)
        if __debug__ and not hasattr(clusters, "__iter__"):
            raise TypeError("`orbpairs` must iterable.")
        dict_cluster_ind = {}
        for i, cluster in enumerate(clusters):
            orbpair_nodup = set(cluster)
            if __debug__:
                if not (
                    isinstance(cluster, (list, tuple))
                    and (len(cluster) == 2 or len(cluster) == 4)
                    and all(isinstance(ind, int) for ind in cluster)
                ):
                    raise TypeError("Each orbital cluster must be a 2-tuple/list or 4-tuple/list of integers.")
                if len(orbpair_nodup) != len(cluster):
                    raise ValueError("Orbital cluster of the same orbital is invalid")

            cluster = tuple(cluster)
            # sort orbitals within the cluster
            cluster = tuple(sorted(cluster))
            if __debug__ and cluster in dict_cluster_ind:
                raise ValueError(f"The given orbital clusters have multiple entries of {cluster} with index {dict_cluster_ind[cluster]}.")
            dict_cluster_ind[cluster] = i

        self.dict_cluster_ind = dict_cluster_ind
        self.dict_ind_cluster = {i: cluster for cluster, i in dict_cluster_ind.items()}

    def get_col_ind(self, orbpair: tuple[int]):
        """Get the column index that corresponds to the given orbital pair.

        Parameters
        ----------
        orbpair : 2-tuple of int
            Indices of the orbital pair.

        Returns
        -------
        col_ind : int
            Column index that corresponds to the given orbital pair.

        Raises
        ------
        ValueError
            If given orbital pair is not valid.

        """
        try:
            if isinstance(orbpair, np.ndarray):
                orbpair = tuple(orbpair)
            return self.dict_cluster_ind[orbpair]
        except (KeyError, TypeError):
            raise ValueError(f"Given orbital pair, {orbpair}, is not included in the wavefunction.")

    def get_permutations(self):
        """ Calculate the permutations of indices 0 to nelec.

        Returns
        -------
        perms : list of list of int
            Permutations of the indices (0 to nelec).
        signs : list of int
            Signs of the permutations.
        """

        indices = np.arange(self.nelec, dtype=int)
        perm_list = list(combinations(indices, r=2))

        olp_list = list(combinations(perm_list, r=int(len(indices) / 2)))
        perms = []
        signs = []
        for element in olp_list:
            element_flat = [item for sublist in element for item in sublist]
            no_dup = list(set(element_flat))
            if len(no_dup) == len(indices):
                perms.append(element)
                signs.append(self.get_sign(element))
        return perms, signs

    def get_sign(self, indices: list[int]):
        """Get the sign of the permutation of the given indices.

        Parameters
        ----------
        indices : list of int
            Indices of the orbitals.

        Returns
        -------
        sign : int
            Sign of the permutation of the given indices.

        """
        olp = [item for pair in indices for item in pair]
        sign = 1
        for i in range(len(olp)):
            for j in range(i + 1, len(olp)):
                if olp[i] > olp[j]:
                    sign *= -1
        return sign

    def _olp(self, sd: int):
        """Calculate overlap with Slater determinant.
        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp : float
            Overlap of the Slater determinant with creation CC wavefunction.

        """
        occ_indices = [slater.occ_indices(sd)] * len(self.permutations)
        single_prods = np.fromiter(map(self.calculate_product, occ_indices, self.permutations, self.signs), dtype=float)
        olp = np.sum(single_prods)
        return olp

    def calculate_product(self, occ_indices, permutation, sign):
        """Calculate the product of the parameters of the given permutation.
        
        Parameters
        ----------
        occ_indices : list of int
            Occupation indices of the Slater determinant.
        permutation : list of 2-tuple of int
            Permutation of the orbital pairs.
        sign : int
            Sign of the permutation.
        
        Returns
        -------
        prod : float
            Product of the parameters of the given permutation
        """

        col_inds = list(map(self.get_col_ind, occ_indices.take(permutation)))
        prod = sign * np.prod(self.params[col_inds])
        return prod

    def _olp_deriv(self, sd: int):
        """Calculate the derivative of the overlap
        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        output : np.ndarray
            Overlap of the Slater determinant with the exponential geminal.

        """

        occ_indices = slater.occ_indices(sd)
        output = np.zeros(len(self.params))
        mapped_permutations = (tuple((occ_indices[i], occ_indices[j]) for i, j in perm) for perm in self.permutations)

        for perm in mapped_permutations:
            sign = self.get_sign(perm)
            for pair in perm:
                col_ind = self.get_col_ind(pair)
                output[col_ind] += sign * np.prod([self.params[self.get_col_ind(p)] for p in perm if p != pair])
        return output

    def get_overlap(self, sd: int, deriv=None):
        r"""Return the (derivative) overlap of the wavefunction with a Slater determinant.

        .. math::
           | \Psi \rangle = \sum_{\textbf{m} \in S} \sum_{\{i_1 j_1, ..., i_{n_m} j_{n_m} \} = \textbf{m}} sgn (\sigma(\{i_1 j_1, ..., i_{n_m} j_{n_m} \}))\prod_{k}^{n_m} c_{i_k j_k} | \textbf{m} \rangle
        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : I am confused about this
            whether to calculate the derivative or not. Default: None
            currently it can only calculate derivative w.r.t. all params

        Returns
        -------
        overlap : {float, np.ndarray}
            Overlap (or derivative of the overlap) of the wavefunction with the given Slater
            determinant.

        """

        if deriv is None:
            return self._olp(sd)
        else:
            return self._olp_deriv(sd)[deriv]
