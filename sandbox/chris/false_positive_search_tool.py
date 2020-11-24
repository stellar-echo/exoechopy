import numpy as np
from scipy.ndimage import morphology


def find_nonflaring_regions(lightcurve, known_flares, forwardpad, backpad=2, dilation_iter=1):
    """Identify non-flaring regions for false-alarm testing

    If you plan to search for echoes with 6-sigma flares, say,
     then you may want to use a list of 4-sigma flares to seed this function
     to avoid picking up minor, though still significant, flares

    Parameters
    ----------
    lightcurve
        Flux value array
    known_flares
        Indices of known flares, these will be masked out
    forwardpad
        Indices before known flares to mask out
    backpad
        Indices after known flares to mask out
    dilation_iter
        Number of dilation iterations to perform on the nan's from the base lightcurve

    Returns
    -------
    indices of non-flaring regions, mask of non-flaring regions
    """
    # Rejected regions are areas where we do not allow a flare to be injected
    # Reject nans:
    rejected_regions = np.isnan(lightcurve)
    # Pad the region around nans by dilation_iter:
    rejected_regions = morphology.binary_dilation(rejected_regions, iterations=dilation_iter)
    # Reject regions already covered by our lightcurve:
    for flare in known_flares:
        # Make sure existing flaring regions will not end up in the tails, either:
        start_ind = max(0, flare - backpad - forwardpad)
        end_ind = min(flare + forwardpad + backpad, len(lightcurve))
        rejected_regions[start_ind:end_ind] = True
    return np.nonzero(~rejected_regions)[0], ~rejected_regions


def select_flare_injection_sites(lightcurve, candidate_indices,
                                 num_flares, forwardpad, backpad=2, num_reselections=1,
                                 trial_timeout=100):
    """

    Parameters
    ----------
    lightcurve
        Original lightcurve
    candidate_indices
        List of available indices to select from
    num_flares
        Number of flares to select
    forwardpad
        Number of indices after a flare to mask out
    backpad
        Number of indices before a flare to mask out
    num_reselections
        Number of collections to produce (does not currently check if collections are different from each other!)
    trial_timeout
        If no collection of flares is found after this many iterations, break
        This can happen if, for example, there are 40 available indices, but they are all adjacent so
        the forwardpad term will mask them all from a single flare selection.
        Typically failure will be more subtle than that, so we just use a timeout.

    Returns
    -------
    injection_remixes
        List of np.ndarray of selected indices that can be used for flare injection studies
    """
    injection_remixes = []
    if len(candidate_indices) < num_flares:
        raise AttributeError("More flares requested than available indices to select from")
    for _i in range(num_reselections):
        reset_ct = 0
        while reset_ct < trial_timeout:
            available_lc = np.zeros_like(lightcurve, dtype=bool)
            available_lc[candidate_indices] = True
            remaining_candidates = np.copy(candidate_indices)
            index_selections = []
            success = True
            for f_i in range(num_flares):
                # If no remaining locations, try again:
                if not np.any(available_lc):
                    success = False
                    break
                new_index = np.random.choice(remaining_candidates)
                index_selections.append(new_index)
                start_ind = max(0, new_index - backpad)
                end_ind = min(new_index + forwardpad, len(lightcurve))
                available_lc[start_ind:end_ind] = False
                remaining_candidates = np.nonzero(available_lc)[0]
            if not success:
                reset_ct += 1
                continue
            else:
                break
        if reset_ct >= trial_timeout:
            raise ValueError("Unable to identify a solution")
        injection_remixes.append(np.array(index_selections))
    return injection_remixes


def detect_disk(*args):
    """Stand-in function for psuedocode example"""
    pass


if __name__ == '__main__':
    # Num points in my fake lightcurve
    num_indices = 10000
    # Num flares to use in my fake lightcurve
    num_flares = 25
    # Generate fake flux for demo purposes, would otherwise use the flattened LC
    my_flux = np.random.random(num_indices)
    my_flares = np.random.choice(num_indices, size=num_flares)
    my_flux[my_flares] = 10
    # Number of times you want to search for false positives (should be a fairly large number in practice):
    num_flare_reinjection_studies = 20
    # Some representative thing that is meaningful to some analysis, but here is just pseudocode:
    detection_threshold = 0.1

    # !NOTE!  You should seed the find_nonflaring_regions() with a
    # lower sigma-threshold than your echo search in order to overexclude flares,
    # otherwise you could end up with small flares inside your selected regions.
    # E.g., if you are using 6-sigma flares, use 4.5-sigma detection to seed find_nonflaring_regions()

    # Find all viable indices that are outside of nan and flaring regions:
    nonflaring_indices, nonflaring_mask = find_nonflaring_regions(my_flux, my_flares, forwardpad=40, backpad=2)
    # From all viable indices, select num_reselections specific flare locations
    all_nonflare_selections = select_flare_injection_sites(my_flux, nonflaring_indices, num_flares,
                                                           forwardpad=40, backpad=2,
                                                           num_reselections=num_flare_reinjection_studies)
    # Iterate through the various re-selections:
    all_results = []
    for nonflare_indices in all_nonflare_selections:
        # Generate a copy of the lightcurve that can be tampered with without damaging the real data:
        temp_new_flux = np.copy(my_flux)
        # Inject your real flares into these non-flaring regions:
        for f_i, flare_index in enumerate(nonflare_indices):
            # Inject the real flare into the nonflaring region
            # (I actually grab this index and +/-1 index on long-cadence data)
            temp_new_flux[flare_index] = my_flux[my_flares[f_i]]
        # Re-perform your detection analysis on this new lightcurve
        # Example:
        disk_detection_result = []
        for orientation in range(0, 91):
            for radius in 10**np.linspace(0.1, 4, 20):
                disk_detection_result.append(detect_disk(temp_new_flux, nonflare_indices))
        all_results.append(disk_detection_result)

    # Find out if you accidentally found echoes where there shouldn't be, and if so, how often.
    # This may be sensitive to disk orientation/radius depending on how your code runs,
    # so you may have a vector/matrix result
    num_false_hits = 0
    for result in all_results:
        # Did you find any disks in the nonflaring regions?
        if result > detection_threshold:
            num_false_hits += 1
    # Here's the entire point of this whole code:
    false_positve_rate = num_false_hits/num_flare_reinjection_studies

    # If you performed this as a function of some parameter, like disk radius,
    # you can build up a histogram or more detailed analysis
    # to show which detection regimes are more sensitive, for instance.
    # If you get too many false positives, you may want to adjust confidence intervals/detection_threshold/etc.

