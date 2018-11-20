
from astropy import units as u
import numpy as np
from ..utils import *

__all__ = ['OrbitSearch']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class OrbitSearch:
    def __init__(self,
                 flare_catalog: 'FlareCatalog',
                 dummy_orbiter: 'KeplerianOrbit',
                 cadence: u.Quantity,
                 lag_offset: int=0,
                 clip_range: tuple=(0, None)):
        """Class for searching orbits for echoes

        Parameters
        ----------
        flare_catalog
            A pre-processed collection of flares
        dummy_orbiter
            Object to apply orbital tests to
        cadence
            Sample cadence (required to correctly align lag times with indices)
        lag_offset
            Optional lag offset index
            Some flare_catalogs have a fixed index offset (e.g., caused by convolution filters that clip end points)
        clip_range
            Optional lag index clipping to prevent overrun
        """
        self._flare_catalog = flare_catalog
        self._flare_times = flare_catalog.get_flare_times().to(u.s).value

        self._dummy_orbiter = dummy_orbiter

        self._cadence = cadence.to(u.s).value
        self._lag_offset = lag_offset
        self._clip_range = clip_range

        self._cache = {}

    def run(self,
            earth_direction_vector=None,
            lag_func=np.mean,
            num_interpolation_points: int=50,
            resample_order: np.ndarray=None,
            **kwargs):
        """
        Search over single or multi-dimensional orbit space for echoes

        Parameters
        ----------
        earth_direction_vector
            Vector or np.array of vectors
            If an array of vectors is provided, will search over the vectors as a parameter
        lag_func
            Function to apply to the list of aligned lags (typically np.mean or np.sum)
        num_interpolation_points
            Number of points to interpolate for the orbital positions
        resample_order
            Optional array of resampled indices to pass to run_lag_hypothesis
        kwargs
            Dictionary of search parameters
            e.g.,
            {'inclination': u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad')}

        Returns
        -------
        np.ndarray
        """
        # Verify keys all exist
        # (if there's a typo, would silently create a new variable if we didn't do this):
        for k in kwargs.keys():
            _ = getattr(self._dummy_orbiter, k)

        # Create the search space
        keys = []
        search_parameters = []
        # See if there are multiple earth orientations requested (important to be first in list):
        if earth_direction_vector.ndim > 1:
            keys.append('earth_direction_vector')
            search_parameters.append(earth_direction_vector)
        for k, v in kwargs.items():
            try:
                # Is this a search parameter?
                _ = iter(v)
                keys.append(k)
                search_parameters.append(v)
            except TypeError:
                # Not iterated, just setting a value:
                setattr(self._dummy_orbiter, k, v)

        # Perform the search:
        if resample_order is None:
            return_array = np.zeros([len(x) for x in search_parameters])
        else:
            dim_list = [len(resample_order)]
            dim_list.extend([len(x) for x in search_parameters])
            return_array = np.zeros(dim_list)

        # Magic happens here:
        def recursive_search(depth, ind, max_depth):
            if depth < max_depth:
                _k = keys[depth]
                for _v_i, _val in enumerate(search_parameters[depth]):
                    setattr(self._dummy_orbiter, _k, _val)
                    ind[depth] = _v_i
                    recursive_search(depth+1, ind, max_depth)
            else:
                planet_vectors = self._dummy_orbiter.evaluate_positions_at_times_lw(self._flare_times,
                                                                                    num_points=num_interpolation_points)
                # Subtract the filter_width, since that operation offset the lags:
                all_lags = compute_lag_simple(planet_vectors,
                                              earth_direction_vector) / self._cadence + self._lag_offset
                # Clip to avoid running over array size: (note, also should de-weight these points)
                all_lags = np.clip(np.round(all_lags), self._clip_range[0], self._clip_range[1]).astype('int')
                if resample_order is None:
                    return_array[tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags, func=lag_func)
                else:
                    for n_i, new_ordering in enumerate(resample_order):
                        return_array[n_i, tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                               func=lag_func,
                                                                                               resample_order=new_ordering)

        # Manually iterate through the highest level, since it could be the earth_direction_vector,
        # which requires special treatment:
        k = keys[0]
        max_depth = len(keys)
        if resample_order is None:
            index = np.zeros(return_array.ndim, dtype='int')
        else:
            index = np.zeros(return_array.ndim-1, dtype='int')
        for v_i, val in enumerate(search_parameters[0]):
            if k != 'earth_direction_vector':
                setattr(self._dummy_orbiter, k, val)
            else:
                earth_direction_vector = val
            index[0] = v_i
            recursive_search(1, index, max_depth)

        return return_array, keys
