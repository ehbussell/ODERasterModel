"""Tools to calculate likelihood and fits of raster model parameters, given simulation data."""

import copy
import numpy as np
import pymc3 as pm
import theano
from scipy.optimize import minimize
from IndividualSimulator.utilities import output_data
import raster_tools

def fit_raster_MCMC(data_stub, kernel_generator, kernel_params, target_raster, nsims=None,
                    mcmc_params=None, likelihood_func=None):
    """Run MCMC to find parameter distributions for raster model.

    Required arguments:
        data_stub:          Simulation data stub to use.
        kernel_generator:   Function that generates kernel function for given kernel parameters.
        kernel_params:      Values of kernel parameters to use. List where each element corresponds
                            to a single kernel parameter, each entry is (name, value). Each value
                            can be a fixed number, or a tuple specifying bounds on that parameter
                            for a uniform prior.
        target_raster:      Raster header dictionary for target raster description of simulation
                            output.

    Optional arguments:
        nsims:              Number of simulations from data to use.  If set to None (default) then
                            all runs in the simulation output will be used.
        mcmc_params:        Dictionary of tuning parameters to use for MCMC run.  Contains:
                                iters:  Number of iterations in MCMC routine (default 1000).
        likelihood_func:    Precomputed log likelihood function (LikelihoodFunction class). If None,
                            then this is generated.
    """

    # TODO adjust to use full/partial precomputation depending on number of cells

    if likelihood_func is None:
        print("Precomputing likelihood function")
        likelihood_func = precompute_loglik(data_stub, nsims, target_raster, end_time=None,
                                            ignore_outside_raster=True, precompute_level="full")

    if mcmc_params is None:
        mcmc_params = {'iters':1000}

    basic_model = pm.Model()

    with basic_model:
        param_dists = []

        for name, val in kernel_params:
            if isinstance(val, tuple) and len(val) == 2:
                param_dists.append(pm.Uniform(name, lower=val[0], upper=val[1]))
            elif isinstance(val, (int, float)):
                param_dists.append(val)
            else:
                raise ValueError("Invalid kernel parameter options!")

        kernel_vals = kernel_generator(*param_dists)(likelihood_func.distances)
        params = theano.tensor.concatenate([[1.0], kernel_vals])

        if likelihood_func.precompute_level == "full":
            log_likelihood = likelihood_func.get_function(params)
        elif likelihood_func.precompute_level == "partial":
            log_likelihood = likelihood_func.get_function(
                params, kernel_generator(*param_dists)(likelihood_func.rel_pos_array[:, :]))

        y_obs = pm.DensityDist('likelihood', log_likelihood, observed=-9999)

        start = pm.find_MAP()
        print(start)
        # trace = pm.sample(mcmc_params['iters'], progressbar=True, step=pm.Metropolis(), start=start)
        trace = pm.sample(mcmc_params['iters'], progressbar=True, start=start)

    return trace


def fit_raster_MLE(data_stub, kernel_generator, kernel_params, target_raster, nsims=None,
                   likelihood_func=None, precompute_level="full", kernel_jac=None, use_theano=True):
    """Maximise likelihood to find parameter distributions for raster model.

    Required arguments:
        data_stub:          Simulation data stub to use.
        kernel_generator:   Function that generates kernel function for given kernel parameters.
        kernel_params:      Values of kernel parameters to use. List where each element corresponds
                            to a single kernel parameter, each entry is (name, value). Each value
                            can be a fixed number, or a tuple specifying bounds on that parameter
                            for a uniform prior.
        target_raster:      Raster header dictionary for target raster description of simulation
                            output.

    Optional arguments:
        nsims:              Number of simulations from data to use.  If set to None (default) then
                            all runs in the simulation output will be used.
        likelihood_func:    Precomputed log likelihood function (LikelihoodFunction class). If None,
                            then this is generated.
        precompute_level:   If likelihood_func is None, and therefore the precomputed log likelihood
                            function is to be generated, then this level of precomputation is used.
                            Options are full and partial.
    """

    # TODO adjust to use full/partial precomputation depending on number of cells

    if likelihood_func is None:
        print("Precomputing likelihood function")
        likelihood_func = precompute_loglik(
            data_stub, nsims, target_raster, end_time=None, ignore_outside_raster=True,
            precompute_level=precompute_level)
        print("Completed")

    if use_theano:

        basic_model = pm.Model()

        with basic_model:
            param_dists = []

            for name, val in kernel_params:
                if isinstance(val, tuple) and len(val) == 2:
                    param_dists.append(pm.Uniform(name, lower=val[0], upper=val[1]))
                elif isinstance(val, (int, float)):
                    param_dists.append(val)
                else:
                    raise ValueError("Invalid kernel parameter options!")

            kernel_vals = kernel_generator(*param_dists)(likelihood_func.distances)
            params = theano.tensor.concatenate([[1.0], kernel_vals])

            if likelihood_func.precompute_level == "full":
                log_likelihood = likelihood_func.get_function(params)
            elif likelihood_func.precompute_level == "partial":
                log_likelihood = likelihood_func.get_function(
                    params, kernel_generator(*param_dists)(likelihood_func.rel_pos_array[:, :]))

            y_obs = pm.DensityDist('likelihood', log_likelihood, observed=-9999)

            start = pm.find_MAP(return_raw=True)

        return start

    else:

        x0 = [0, 0]

        bounds = [val for name, val in kernel_params]

        def neg_loglik(params):
            # First reverse logit transform parameters
            _params_transformed = np.array(
                [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, params)])
            val, jacobian = likelihood_func.eval_loglik(
                kernel_generator(*_params_transformed), jac=kernel_jac(*_params_transformed))

            if jacobian is not None:
                scale_transform = np.array([
                    (x-a)*(b-x)/(b-a) for (a, b), x in zip(bounds, _params_transformed)])
                _jacobian_transformed = jacobian * scale_transform
                return (np.nan_to_num(-val), np.nan_to_num(-_jacobian_transformed))

            return np.nan_to_num(-val)

        param_fit = minimize(neg_loglik, x0, jac=True, method="L-BFGS-B")
        x1 = param_fit.x
        x2 = np.array(
            [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, x1)])

        return x2

def precompute_loglik(data_stub, nsims, raster_header, end_time=None,
                      ignore_outside_raster=False, output_freq=10, precompute_level="full"):
    """Generate log likelihood function given simulation data.

    If precompute_level is 'full': constant factors and full matrix are calculated. This can be slow
        and require large amounts of memory for large grid sizes, but will minimise likelihood
        calculation times.

    If precompute_level is 'partial': constant factors are still calculated, and function is setup
        to calculate the likelihood on-the-fly, recalculating infectious pressures at each event.
        This will be slower but requires less precalculation and memory.
    """

    if precompute_level == "full":
        return _precompute_full(data_stub, nsims, raster_header, end_time, ignore_outside_raster,
                                output_freq)
    elif precompute_level == "partial":
        return _precompute_partial(data_stub, nsims, raster_header, end_time, ignore_outside_raster,
                                   output_freq)
    else:
        raise ValueError("Unrecognised precompute level!")


def _precompute_full(data_stub, nsims, raster_header, end_time=None,
                     ignore_outside_raster=False, output_freq=10):
    """Fully precompute the raster likelihood function."""

    # TODO tidy up this function - maybe use create_cell_data from simulator utilities

    nrows = raster_header['nrows']
    ncols = raster_header['ncols']

    # Define relative position function
    def get_rel_pos(i, j):

        i_pos = np.unravel_index(i, (nrows, ncols))
        j_pos = np.unravel_index(j, (nrows, ncols))
        rel_pos = (abs(i_pos[0] - j_pos[0]), abs(i_pos[1] - j_pos[1]))

        ret_val = rel_pos[1] + (rel_pos[0] * ncols)

        return ret_val

    kernel_length = raster_header['nrows'] * raster_header['ncols']

    const_factors = np.zeros(1+kernel_length)
    matrix_factors = [np.zeros((1+kernel_length))]
    matrix_factors[0][0] = 1

    # Get data
    data = output_data.extract_output_data(data_stub)

    if nsims is None:
        nsims = len(data)

    host_map = {}
    initial_state = np.zeros((kernel_length, 2))

    # Construct initial state
    for index, host in data[0]['host_data'].iterrows():
        # find cell index
        cell = raster_tools.find_position_in_raster(
            (host['posX'], host['posY']), raster_header=raster_header, index=True)
        if cell == -1:
            if ignore_outside_raster:
                continue
            else:
                raise ValueError("Host not in raster!")
        host_map[host['hostID']] = cell
        state = host['initial_state']
        if state == "S":
            initial_state[cell, 0] += 1
        elif state == "I":
            initial_state[cell, 1] += 1
        else:
            raise ValueError("Not S or I!")

    print("Finished cell map and initial state")

    rel_pos_array = np.zeros((kernel_length, kernel_length), dtype=int)
    init_change_term = np.zeros(1+kernel_length)
    for i in range(kernel_length):
        for j in range(kernel_length):
            rel_pos_array[i, j] = get_rel_pos(i, j)
            init_change_term[1+rel_pos_array[i, j]] -= initial_state[j, 1]*initial_state[i, 0]

    print("Finished relative position map and initial change term")

    for x in range(nsims):
        # For each simulation run
        sim = data[x]['event_data']
        next_output = int(len(sim)*(output_freq/100))
        previous_time = 0.0
        state = copy.copy(initial_state)
        change_term = copy.copy(init_change_term)
        all_times = sim['time'].values
        all_host_ids = sim['hostID'].values
        for i in range(len(sim)):
            # For each event
            if all_host_ids[i] in host_map:
                new_row = np.zeros(1+kernel_length)
                inf_cell = host_map[all_host_ids[i]]
                time = all_times[i]
                const_factors += change_term * (time - previous_time)
                const_factors[0] += np.log(state[inf_cell, 0])
                for j in range(kernel_length):
                    rel_pos = rel_pos_array[inf_cell, j]
                    new_row[1+rel_pos] += state[j, 1]
                    if j != inf_cell:
                        change_term[1+rel_pos_array[inf_cell, j]] += state[j, 1]
                        change_term[1+rel_pos_array[j, inf_cell]] -= state[j, 0]
                    else:
                        change_term[1+rel_pos_array[j, j]] += 1 + state[j, 1] - state[j, 0]
                if np.all(new_row == 0):
                    raise ValueError("No entries in matrix row!")
                matrix_factors.append(new_row)
                previous_time = time
                state[inf_cell, 0] -= 1
                state[inf_cell, 1] += 1
            if i >= next_output:
                print("{0}% of events complete".format(int(100*(i+1)/len(sim))))
                next_output += int(len(sim)*(output_freq/100))
        i = len(sim)
        if end_time is not None:
            for j in range(kernel_length):
                for k in range(kernel_length):
                    rel_pos = rel_pos_array[j, k]
                    const_factors[1+rel_pos] -= state[k, 1]*state[j, 0]*(end_time - previous_time)

    matrix = np.array(matrix_factors)

    positions = [np.unravel_index(x, (nrows, ncols)) for x in range(kernel_length)]
    distances = np.array([np.sqrt(x*x + y*y) for x, y in positions])

    ret_data = {
        "const_factors": const_factors,
        "matrix": matrix,
        "distances": distances
    }

    return LikelihoodFunction(precompute_level="full", data=ret_data)


def _precompute_partial(data_stub, nsims, raster_header, end_time=None,
                        ignore_outside_raster=False, output_freq=10):
    """Partially precompute the raster likelihood function."""

    # TODO tidy up this function - maybe use create_cell_data from simulator utilities

    nrows = raster_header['nrows']
    ncols = raster_header['ncols']

    # Define relative position function
    def get_rel_pos(i, j):

        i_pos = np.unravel_index(i, (nrows, ncols))
        j_pos = np.unravel_index(j, (nrows, ncols))
        rel_pos = (abs(i_pos[0] - j_pos[0]), abs(i_pos[1] - j_pos[1]))

        ret_val = rel_pos[1] + (rel_pos[0] * ncols)

        return ret_val

    kernel_length = raster_header['nrows'] * raster_header['ncols']

    const_factors = np.zeros(1+kernel_length)

    # Get data
    data = output_data.extract_output_data(data_stub)

    if nsims is None:
        nsims = len(data)

    host_map = {}
    initial_state = np.zeros((kernel_length, 2))

    # Construct initial state
    for index, host in data[0]['host_data'].iterrows():
        # find cell index
        cell = raster_tools.find_position_in_raster(
            (host['posX'], host['posY']), raster_header=raster_header, index=True)
        if cell == -1:
            if ignore_outside_raster:
                continue
            else:
                raise ValueError("Host not in raster!")
        host_map[host['hostID']] = cell
        state = host['initial_state']
        if state == "S":
            initial_state[cell, 0] += 1
        elif state == "I":
            initial_state[cell, 1] += 1
        else:
            raise ValueError("Not S or I!")

    print("Finished cell map and initial state")

    rel_pos_array = np.zeros((kernel_length, kernel_length), dtype=int)
    init_change_term = np.zeros(1+kernel_length)
    for i in range(kernel_length):
        for j in range(kernel_length):
            rel_pos_array[i, j] = get_rel_pos(i, j)
            init_change_term[1+rel_pos_array[i, j]] -= initial_state[j, 1]*initial_state[i, 0]

    print("Finished relative position map and initial change term")

    all_inf_cell_ids = []

    for x in range(nsims):
        # For each simulation run
        inf_cell_ids = []
        sim = data[x]['event_data']
        next_output = int(len(sim)*(output_freq/100))
        previous_time = 0.0
        state = copy.copy(initial_state)
        change_term = copy.copy(init_change_term)
        all_times = sim['time'].values
        all_host_ids = sim['hostID'].values
        for i in range(len(sim)):
            # For each event
            if all_host_ids[i] in host_map:
                inf_cell = host_map[all_host_ids[i]]
                inf_cell_ids.append(inf_cell)
                time = all_times[i]
                const_factors += change_term * (time - previous_time)
                const_factors[0] += np.log(state[inf_cell, 0])
                for j in range(kernel_length):
                    rel_pos = rel_pos_array[inf_cell, j]
                    if j != inf_cell:
                        change_term[1+rel_pos_array[inf_cell, j]] += state[j, 1]
                        change_term[1+rel_pos_array[j, inf_cell]] -= state[j, 0]
                    else:
                        change_term[1+rel_pos_array[j, j]] += 1 + state[j, 1] - state[j, 0]
                previous_time = time
                state[inf_cell, 0] -= 1
                state[inf_cell, 1] += 1
            if i >= next_output:
                print("{0}% of events complete".format(int(100*(i+1)/len(sim))))
                next_output += int(len(sim)*(output_freq/100))
        i = len(sim)
        if end_time is not None:
            for j in range(kernel_length):
                for k in range(kernel_length):
                    rel_pos = rel_pos_array[j, k]
                    const_factors[1+rel_pos] -= state[k, 1]*state[j, 0]*(end_time - previous_time)
        all_inf_cell_ids.append(np.array(inf_cell_ids))

    positions = [np.unravel_index(x, (nrows, ncols)) for x in range(kernel_length)]
    distances = np.array([np.sqrt(x*x + y*y) for x, y in positions])

    ret_data = {
        "distances": distances,
        "const_factors": const_factors,
        "all_inf_cell_ids": all_inf_cell_ids,
        "initial_inf": initial_state[:, 1],
        "rel_pos_array": rel_pos_array
    }

    return LikelihoodFunction(precompute_level="partial", data=ret_data)


class LikelihoodFunction:
    """Class holding precomputed likelihood information."""
    # TODO implement partial computation

    def __init__(self, precompute_level, data):
        if precompute_level == "full":
            self.precompute_level = "full"
            self.const_factors = data['const_factors']
            self.matrix = data['matrix']
            self.distances = data['distances']

        elif precompute_level == "partial":
            self.precompute_level = "partial"
            self.const_factors = data['const_factors']
            self.distances = data['distances']
            self.all_inf_cell_ids = data['all_inf_cell_ids']
            self.initial_inf = data['initial_inf']
            self.rel_pos_array = data['rel_pos_array']

        else:
            raise ValueError("Unrecognised precomputation level!")

    @classmethod
    def from_file(cls, filename):
        """Initialise from .npz file."""

        loaded = np.load(filename)
        return cls(loaded['level'], loaded)

    def eval_loglik(self, kernel, jac=None):
        """Compute the data likelihood for the given kernel function."""

        if self.precompute_level == "full":
            kernel_vals = np.concatenate([[1.0], kernel(self.distances)])

            log_lik = np.dot(self.const_factors, kernel_vals)
            dot = np.dot(self.matrix, kernel_vals)
            log_lik += np.sum(np.log(dot))

            if jac is not None:
                dk = jac(self.distances).T
                jacobian = np.dot(self.const_factors[1:], dk)
                jacobian += np.array([
                    np.sum(np.dot(self.matrix[:, 1:], dk[:, i]) / dot) for i in range(dk.shape[1])])
            else:
                jacobian = None
            
            return (log_lik, jacobian)

        elif self.precompute_level == "partial":
            kernel_vals = np.concatenate([[1.0], kernel(self.distances)])

            log_lik = np.sum(np.dot(self.const_factors, kernel_vals))

            print("Constant factors complete")

            full_kernel = kernel(self.rel_pos_array)
            for inf_cell_ids in self.all_inf_cell_ids:
                inf_state = copy.copy(self.initial_inf)
                print("Copied infected cells")
                for inf_cell in inf_cell_ids:
                    log_lik += np.log(np.sum(np.dot(inf_state, full_kernel[:, inf_cell])))
                    inf_state[inf_cell] += 1
                print("Run complete")

            return log_lik

        else:
            raise ValueError("Unrecognised precomputation level!")

    def get_function(self, params, full_kernel=None):
        """Get log likelihood function for use with pymc3."""

        if self.precompute_level == "full":
            log_lik = theano.tensor.sum(theano.tensor.dot(
                self.const_factors, params)) + theano.tensor.sum(theano.tensor.log(
                    theano.tensor.dot(self.matrix, params)))

            def log_likelihood(required_argument):
                return log_lik

            return log_likelihood

        elif self.precompute_level == "partial":
            # TODO Handle multiple runs
            if len(self.all_inf_cell_ids) != 1:
                raise NotImplementedError("Partial precompute cannot handle multiple runs!")

            log_lik = theano.tensor.sum(theano.tensor.dot(self.const_factors, params))

            # Scan over events
            ([inf_state, lik_terms], updates) = theano.scan(
                fn=self._inf_cell, sequences=self.all_inf_cell_ids[0],
                outputs_info=[dict(initial=copy.copy(self.initial_inf), taps=[-1]), None],
                non_sequences=[full_kernel], strict=True)
            # Combine results
            log_lik += theano.tensor.sum(theano.tensor.log(lik_terms))

            def log_likelihood(required_argument):
                return log_lik

            return log_likelihood

        else:
            raise ValueError("Unrecognised precomputation level!")

    def save(self, savefile, identifier=None):
        """Save likelihood precalculation to .npz file."""

        if self.precompute_level == "full":
            np.savez_compressed(savefile, level="full", identifier=identifier,
                                distances=self.distances, const_factors=self.const_factors,
                                matrix=self.matrix)

        elif self.precompute_level == "partial":
            np.savez_compressed(savefile, level="partial", identifier=identifier,
                                distances=self.distances, const_factors=self.const_factors,
                                all_inf_cell_ids=self.all_inf_cell_ids,
                                initial_inf=self.initial_inf, rel_pos_array=self.rel_pos_array)

        else:
            raise ValueError("Unrecognised precomputation level!")


    def _inf_cell(self, inf_id, inf_state_m1, full_kernel):

        lik_term = theano.tensor.sum(theano.tensor.dot(inf_state_m1, full_kernel[:, inf_id]))

        new_inf_state = theano.tensor.set_subtensor(inf_state_m1[inf_id], inf_state_m1[inf_id]+1)

        return [new_inf_state, lik_term]
