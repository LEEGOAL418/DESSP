import numpy as np
import random
from ase import Atoms
from scipy.spatial import cKDTree


class BaseOperator:
    """
    Base class for all custom genetic operators.
    Standardizes offspring creation and ensures .info/data/key_value_pairs are complete.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def initialize_offspring(self, parents, offspring):
        """Initializes offspring.info, ensuring parent information is included."""
        parent_ids = [p.info.get('confid', 'N/A') for p in parents]

        offspring.info['data'] = {'parents': parent_ids}
        offspring.info['key_value_pairs'] = {}

        return offspring

    def sanity_check(self, atoms):
        """Checks if C atoms exist to prevent generating invalid configurations."""
        nC = np.sum(atoms.get_atomic_numbers() == 6)
        if nC == 0:
            return False
        return True


class SiteBasedCrossoverOperator(BaseOperator):
    def __init__(self, slab, all_ht_sites, n_li, verbose=False):
        super().__init__(verbose=verbose)
        self.slab = slab  # Should always be the graphene substrate without Li
        self.all_ht_sites = np.array(all_ht_sites)
        self.n_li = n_li
        self.kdtree = cKDTree(self.all_ht_sites)

    def get_new_individual(self, parents):
        offspring_core = self._operate(parents)
        if offspring_core is None or not self.sanity_check(offspring_core):
            return None, getattr(self, "failure_desc", "Invalid offspring")

        offspring = self.initialize_offspring(parents, offspring_core)
        offspring.info['data']['li_indices'] = self.child_indices
        desc = f"Crossover:from_{offspring.info['data']['parents'][0]}&{offspring.info['data']['parents'][1]}"
        return offspring, desc

    def _operate(self, parents):
        parent1, parent2 = parents
        indices1 = self._get_indices_from_parent(parent1)
        indices2 = self._get_indices_from_parent(parent2)

        if not indices1 or not indices2:
            self.failure_desc = "Crossover:Failed_Get_Indices"
            return None

        combined_indices = list(indices1.union(indices2))
        if len(combined_indices) < self.n_li:
            self.failure_desc = "Crossover:Not_Enough_Sites"
            return None

        child_indices_arr = np.sort(
            np.random.choice(combined_indices, self.n_li, replace=False)
        )
        self.child_indices = child_indices_arr.tolist()

        child_positions = self.all_ht_sites[child_indices_arr]
        li_atoms = Atoms("Li" * self.n_li, positions=child_positions)

        offspring_core = self.slab.copy()
        offspring_core.extend(li_atoms)
        offspring_core.wrap()
        return offspring_core

    def _get_indices_from_parent(self, parent):
        try:
            indices = parent.info['data']['li_indices']
            if indices:
                return set(indices)
        except (KeyError, TypeError):
            pass
        li_positions = parent.get_positions()[parent.get_atomic_numbers() == 3]
        if len(li_positions) == 0:
            return set()
        _, indices = self.kdtree.query(li_positions, k=1)
        return set(indices)


class LiVacancySwapOperator(BaseOperator):
    def __init__(self, all_ht_sites, verbose=False):
        super().__init__(verbose=verbose)
        self.all_ht_sites = np.array(all_ht_sites)
        # Added to handle external files without metadata
        self.kdtree = cKDTree(self.all_ht_sites)

    def get_new_individual(self, parents):
        offspring_core = self._operate(parents[0])
        if offspring_core is None or not self.sanity_check(offspring_core):
            return None, getattr(self, "failure_desc", "Invalid offspring")

        offspring = self.initialize_offspring([parents[0]], offspring_core)
        offspring.info['data']['li_indices'] = self.new_indices
        return offspring, self.success_desc

    def _operate(self, parent):
        atoms = parent.copy()

        # HT indices currently occupied by Li (from info)
        occupied_indices = self._get_indices_from_parent(atoms)
        vacant_indices = set(range(len(self.all_ht_sites))) - occupied_indices

        if not occupied_indices or not vacant_indices:
            self.failure_desc = "LiVacancySwap:No_Swap_Possible"
            return None

        # Randomly select one Li site to swap with a vacant site
        idx_to_remove = random.choice(list(occupied_indices))
        idx_to_add = random.choice(list(vacant_indices))

        # Locate corresponding Li index from info
        li_indices = atoms.info.get("data", {}).get("li_indices", [])
        if not li_indices or idx_to_remove not in li_indices:
            self.failure_desc = "LiVacancySwap:li_indices missing or mismatch"
            return None

        # Find the atom index of the Li (index of the Li atom in positions)
        li_atom_indices = [i for i, Z in enumerate(atoms.get_atomic_numbers()) if Z == 3]
        try:
            li_to_move = li_atom_indices[li_indices.index(idx_to_remove)]
        except ValueError:
            self.failure_desc = "LiVacancySwap:Failed_to_Map"
            return None

        # Update Li coordinates to the new vacant site
        atoms.positions[li_to_move] = self.all_ht_sites[idx_to_add]
        atoms.wrap()

        # Update indices
        self.new_indices = sorted((set(li_indices) - {idx_to_remove}) | {idx_to_add})
        self.success_desc = f"LiVacancySwap:site_{idx_to_remove}_to_{idx_to_add}"
        return atoms

    # Reuses the crossover _get_indices_from_parent method
    _get_indices_from_parent = SiteBasedCrossoverOperator._get_indices_from_parent


class InplaneShiftOperator(BaseOperator):
    def __init__(self, layer='top', max_disp=0.2, verbose=False):
        super().__init__(verbose=verbose)
        self.layer = layer
        self.max_disp = max_disp

    def get_new_individual(self, parents):
        offspring = self._operate(parents[0])
        if offspring is None or not self.sanity_check(offspring):
            return None, "InplaneShift:Invalid offspring"
        desc = f"InplaneShift:{self.layer}_by_({self.dx:.2f},{self.dy:.2f})"
        return offspring, desc

    def _operate(self, parent):
        atoms = parent.copy()
        pos = atoms.get_positions()
        c_mask = (atoms.get_atomic_numbers() == 6)
        if not np.any(c_mask):
            return None
        z_coords_c = pos[c_mask, 2]
        z_median_c = np.median(z_coords_c)
        layer_mask = (pos[:, 2] > z_median_c) if self.layer == 'top' else (pos[:, 2] < z_median_c)
        final_mask = layer_mask & c_mask

        self.dx = random.uniform(-self.max_disp, self.max_disp)
        self.dy = random.uniform(-self.max_disp, self.max_disp)
        disp_vec = np.array([self.dx, self.dy, 0.0])
        pos[final_mask] += disp_vec
        atoms.set_positions(pos)
        atoms.wrap()
        return atoms


class OutplaneStretchOperator(BaseOperator):
    def __init__(self, max_disp=0.15, verbose=False):
        super().__init__(verbose=verbose)
        self.max_disp = max_disp

    def get_new_individual(self, parents):
        offspring = self._operate(parents[0])
        if offspring is None or not self.sanity_check(offspring):
            return None, "OutplaneStretch:Invalid offspring"
        desc = f"OutplaneStretch:by_{self.dz:.3f}"
        return offspring, desc

    def _operate(self, parent):
        atoms = parent.copy()
        pos = atoms.get_positions()
        c_mask = (atoms.get_atomic_numbers() == 6)
        if not np.any(c_mask):
            return None
        z_coords_c = pos[c_mask, 2]
        z_median_c = np.median(z_coords_c)
        top_layer_mask = (pos[:, 2] > z_median_c) & c_mask
        self.dz = random.uniform(-self.max_disp, self.max_disp)
        disp_vec = np.array([0.0, 0.0, self.dz])
        pos[top_layer_mask] += disp_vec
        atoms.set_positions(pos)
        atoms.wrap()
        return atoms