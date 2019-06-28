"""Tools to calculate likelihood and fits of raster model parameters, given simulation data."""

import copy
import pdb
import numpy as np
import pymc3 as pm
import theano
from scipy.optimize import minimize
from IndividualSimulator.utilities import output_data
import raster_tools
from RasterModel import raster_model

def fit_raster_MCMC(kernel_generator, kernel_params, data_stub=None, nsims=None, mcmc_params=None,
                    likelihood_func=None, target_raster=None):
    """Run MCMC to find parameter distributions for raster model.

    Required arguments:
        kernel_generator:   Function that generates kernel function for given kernel parameters.
        kernel_params:      Values of kernel parameters to use. List where each element corresponds
                            to a single kernel parameter, each entry is (name, value). Each value
                            can be a fixed number, or a tuple specifying bounds on that parameter
                            for a uniform prior.

    Optional arguments:
        data_stub:          Simulation data stub to use.
        nsims:              Number of simulations from data to use.  If set to None (default) then
                            all runs in the simulation output will be used.
        mcmc_params:        Dictionary of tuning parameters to use for MCMC run.  Contains:
                                iters:  Number of iterations in MCMC routine (default 1000).
        likelihood_func:    Precomputed log likelihood function (LikelihoodFunction class). If None,
                            then this is generated.
        target_raster:      Raster header dictionary for target raster description of simulation
                            output.
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


def fit_raster_MLE(kernel_generator, kernel_params, param_start=None, data_stub=None, nsims=None,
                   likelihood_func=None, precompute_level="full", target_raster=None,
                   kernel_jac=None, use_theano=False, raw_output=False, primary_rate=False):
    """Maximise likelihood to find parameter distributions for raster model.

    Required arguments:
        kernel_generator:   Function that generates kernel function for given kernel parameters.
        kernel_params:      Values of kernel parameters to use. Dict where each element corresponds
                            to a single kernel parameter, each entry is name: value. Each value
                            can be a fixed number, or a tuple specifying bounds on that parameter
                            for a uniform prior.

    Optional arguments:
        param_start:        Parameter dictionary giving start points for optimisation. If None then
                            starts from centre of prior.
        data_stub:          Simulation data stub to use. Required if likelihood function is not
                            specified.
        nsims:              Number of simulations from data to use. If set to None (default) then
                            all runs in the simulation output will be used. Only required if the
                            likelihood function is not specified.
        likelihood_func:    Precomputed log likelihood function (LikelihoodFunction class). If None,
                            then this is generated.
        precompute_level:   If likelihood_func is None, and therefore the precomputed log likelihood
                            function is to be generated, then this level of precomputation is used.
                            Options are full and partial.
        target_raster:      Raster header dictionary for target raster description of simulation
                            output. Only required if likelihood function is not specified.
        kernel_jac:         Jacobian function of kernel
        primary_rate:       If True then expect PrimaryRate parameter in options. Will then fit
                            primary infection rate.
    """

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

            for name, val in kernel_params.items():
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

            start = pm.find_MAP(return_raw=raw_output)

        return start

    else:

        param_names = sorted(kernel_params)
        if primary_rate:
            param_names.append(param_names.pop(param_names.index("PrimaryRate")))
        bounds = [kernel_params[name] for name in param_names]

        if param_start is None:
            x0 = [0 for name in param_names]
        else:
            x0 = np.array([np.log((param_start[name] - a) / (b - param_start[name]))
                           for name, (a, b) in zip(param_names, bounds)])

        def neg_loglik(params):
            # First reverse logit transform parameters
            _params_transformed = np.array(
                [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, params)])
            if primary_rate:
                val, jacobian = likelihood_func.eval_loglik(
                    kernel_generator(*_params_transformed[:-1]),
                    jac=kernel_jac(*_params_transformed[:-1]), primary_rate=_params_transformed[-1])
            else:
                val, jacobian = likelihood_func.eval_loglik(
                    kernel_generator(*_params_transformed), jac=kernel_jac(*_params_transformed))

            if jacobian is not None:
                scale_transform = np.array([
                    (x-a)*(b-x)/(b-a) for (a, b), x in zip(bounds, _params_transformed)])
                _jacobian_transformed = jacobian * scale_transform
                return (np.nan_to_num(-val), np.nan_to_num(-_jacobian_transformed))

            return np.nan_to_num(-val)

        param_fit = minimize(neg_loglik, x0, jac=True, method="L-BFGS-B", options={'ftol': 1e-12})
        x1 = param_fit.x
        x2 = np.array(
            [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, x1)])

        opt_params = {name: param for name, param in zip(param_names, x2)}

        if raw_output:
            return opt_params, param_fit
        else:
            return opt_params

def fit_raster_SSE(model, kernel_generator, kernel_params, data_stub, param_start=None, nsims=None,
                   target_raster=None, dimensions=None, raw_output=False, primary_rate=False):
    """Minimise summed square errors from simulations to find best fitting parameters.

    Required arguments:
        kernel_generator:   Function that generates kernel function for given kernel parameters.
        kernel_params:      Values of kernel parameters to use. Dict where each element corresponds
                            to a single kernel parameter, each entry is name: value. Each value
                            can be a fixed number, or a tuple specifying bounds on that parameter
                            for a uniform prior.
        data_stub:          Simulation data stub to use.
        param_start:        Parameter dictionary giving start points for optimisation. If None then
                            starts from centre of prior.
        nsims:              Number of simulations from data to use. If set to None (default) then
                            all runs in the simulation output will be used.
        target_raster:      Raster header dictionary for target raster description of simulation
                            output.
        primary_rate:       If True then expect PrimaryRate parameter in options. Will then fit
                            primary infection rate.
    """

    param_names = sorted(kernel_params)
    if primary_rate:
        param_names.append(param_names.pop(param_names.index("PrimaryRate")))
    bounds = [kernel_params[name] for name in param_names]

    if param_start is None:
        x0 = [0 for name in param_names]
    else:
        x0 = np.array([np.log((param_start[name] - a) / (b - param_start[name]))
                       for name, (a, b) in zip(param_names, bounds)])

    times = model.params['times']

    # Extract simulation data at correct resolution
    base_data = output_data.create_cell_data(
        data_stub, target_header=target_raster, ignore_outside_raster=True)
    if nsims is None:
        nsims = len(base_data)
    if dimensions is None:
        dimensions = (target_raster["ncols"], target_raster['nrows'])

    ncells = np.product(dimensions)
    sim_data = np.ndarray((nsims, ncells, len(times)))

    # Extract state at test times for each cell
    for i, dataset in enumerate(base_data):
        for cell in range(ncells):
            current_i = None
            idx = 0
            for t, _, i_state, *_ in dataset[cell]:
                while t > times[idx]:
                    sim_data[i, cell, idx] = current_i
                    idx += 1
                    if idx > len(times):
                        break
                current_i = i_state
            while idx != len(times):
                sim_data[i, cell, idx] = current_i
                idx += 1

    # Setup raster model
    distances = np.zeros((ncells, ncells))
    for i in range(ncells):
        for j in range(ncells):
            dx = abs((i % dimensions[1]) - (j % dimensions[1]))
            dy = abs(int(i/dimensions[1]) - int(j/dimensions[1]))
            distances[i, j] = np.sqrt(dx*dx + dy*dy)

    # Define objective function that sets cell coupling and calculates sum of squared errors
    def objective(params):
        # First reverse logit transform parameters
        _params_transformed = np.array(
            [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, params)])
        if primary_rate:
            kernel = kernel_generator(*_params_transformed[:-1])
            primary_value = _params_transformed[-1]
        else:
            kernel = kernel_generator(*_params_transformed)
            primary_value = 0

        new_coupling = kernel(distances)
        model.params['coupling'] = new_coupling
        model.params['primary_rate'] = primary_value

        no_control_tmp = model.run_scheme(model.no_control_policy)
        no_control_results = np.zeros((ncells, len(times)))
        for cell in range(ncells):
            i_vals = no_control_tmp.results_i["Cell" + str(cell)].values
            no_control_results[cell, :] = i_vals

        sse = 0
        for i, dataset in enumerate(sim_data):
            sse += np.sum(np.square(no_control_results - dataset))

        return sse

    # Minimse SSE
    param_fit = minimize(objective, x0, method="L-BFGS-B", options={'ftol': 1e-12})
    x1 = param_fit.x
    x2 = np.array(
        [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for (a, b), x in zip(bounds, x1)])

    opt_params = {name: param for name, param in zip(param_names, x2)}

    if raw_output:
        return opt_params, param_fit
    else:
        return opt_params


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
    primary_factor = 0.0
    matrix_factors = []

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
    init_prim_term = 0
    for i in range(kernel_length):
        init_prim_term -= initial_state[i, 0]
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
        prim_term = copy.copy(init_prim_term)
        all_times = sim['time'].values
        all_host_ids = sim['hostID'].values
        for i in range(len(sim)):
            # For each event
            if all_host_ids[i] in host_map:
                new_row = np.zeros(1+kernel_length)
                inf_cell = host_map[all_host_ids[i]]
                time = all_times[i]
                const_factors += change_term * (time - previous_time)
                primary_factor += prim_term * (time - previous_time)
                const_factors[0] += np.log(state[inf_cell, 0])
                prim_term -= 1
                new_row[0] = 1
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
                # print("{0}% of events complete".format(int(100*(i+1)/len(sim))))
                next_output += int(len(sim)*(output_freq/100))
        i = len(sim)
        if end_time is not None:
            primary_factor += prim_term * (end_time - previous_time)
            for j in range(kernel_length):
                for k in range(kernel_length):
                    rel_pos = rel_pos_array[j, k]
                    const_factors[1+rel_pos] -= state[k, 1]*state[j, 0]*(end_time - previous_time)

    matrix = np.array(matrix_factors)

    positions = [np.unravel_index(x, (nrows, ncols)) for x in range(kernel_length)]
    distances = np.array([np.sqrt(x*x + y*y) for x, y in positions])

    ret_data = {
        "const_factors": const_factors,
        "primary_factor": primary_factor,
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
    primary_factor = 0.0

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
    dist_array = np.zeros((kernel_length, kernel_length))
    init_change_term = np.zeros(1+kernel_length)
    init_prim_term = 0
    for i in range(kernel_length):
        init_prim_term  -= initial_state[i, 0]
        for j in range(kernel_length):
            rel_pos_array[i, j] = get_rel_pos(i, j)
            init_change_term[1+rel_pos_array[i, j]] -= initial_state[j, 1]*initial_state[i, 0]
            x, y = np.unravel_index(rel_pos_array[i, j], (nrows, ncols))
            dist_array[i, j] = np.sqrt(x*x + y*y)

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
        prim_term = copy.copy(init_prim_term)
        all_times = sim['time'].values
        all_host_ids = sim['hostID'].values
        for i in range(len(sim)):
            # For each event
            if all_host_ids[i] in host_map:
                inf_cell = host_map[all_host_ids[i]]
                inf_cell_ids.append(inf_cell)
                time = all_times[i]
                const_factors += change_term * (time - previous_time)
                primary_factor += prim_term * (time - previous_time)
                const_factors[0] += np.log(state[inf_cell, 0])
                prim_term -= 1
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
                # print("{0}% of events complete".format(int(100*(i+1)/len(sim))))
                next_output += int(len(sim)*(output_freq/100))
        i = len(sim)
        if end_time is not None:
            primary_factor += prim_term * (end_time - previous_time)
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
        "primary_factor": primary_factor,
        "all_inf_cell_ids": all_inf_cell_ids,
        "initial_inf": initial_state[:, 1],
        "rel_pos_array": dist_array
    }

    return LikelihoodFunction(precompute_level="partial", data=ret_data)


class LikelihoodFunction:
    """Class holding precomputed likelihood information."""
    # TODO implement partial computation

    def __init__(self, precompute_level, data):
        if precompute_level == "full":
            self.precompute_level = "full"
            self.const_factors = data['const_factors']
            self.primary_factor = data['primary_factor']
            self.matrix = data['matrix']
            self.distances = data['distances']

        elif precompute_level == "partial":
            self.precompute_level = "partial"
            self.const_factors = data['const_factors']
            self.primary_factor = data['primary_factor']
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

    def eval_loglik(self, kernel, jac=None, primary_rate=None):
        """Compute the data likelihood for the given kernel function.

        Arguments:
            kernel:         Function to evaluate the dispersal kernel from between cell distance
            jac:            Function to evaluate the jacobian of the kernel. If None then the
                            likelihood jacobian is not calculated.
            primary_rate:   Value of primary infection rate. If None then primary infection is not
                            included
        """

        if self.precompute_level == "full":
            if primary_rate is None:
                epsilon = 0.0
            else:
                epsilon = primary_rate

            kernel_vals = np.concatenate([[1.0], kernel(self.distances)])

            log_lik = np.dot(self.const_factors, kernel_vals)

            kernel_vals[0] = epsilon
            dot = np.dot(self.matrix, kernel_vals)
            log_lik += np.sum(np.log(dot))
            log_lik += self.primary_factor * epsilon

            if jac is not None:
                dk = jac(self.distances).T
                dk = np.vstack((np.zeros(dk.shape[1]), dk))
                jacobian = np.dot(self.const_factors, dk)
                if primary_rate is not None:
                    jacobian = np.append(jacobian, self.primary_factor)
                    de = np.zeros((dk.shape[0], 1))
                    de[0] = 1.0
                    dk = np.hstack((dk, de))
                inv_dot = np.reciprocal(dot)
                jacobian += np.dot(inv_dot, np.dot(self.matrix, dk))
            else:
                jacobian = None

            return (log_lik, jacobian)

        elif self.precompute_level == "partial":
            kernel_vals = np.concatenate([[1.0], kernel(self.distances)])

            log_lik = np.dot(self.const_factors, kernel_vals)

            if primary_rate is not None:
                log_lik += self.primary_factor * primary_rate

            if jac is not None:
                dk = jac(self.distances).T
                dk = np.vstack((np.zeros(dk.shape[1]), dk))
                jacobian = np.dot(self.const_factors, dk)
                if primary_rate is not None:
                    jacobian = np.append(jacobian, self.primary_factor)
                full_dk = jac(self.rel_pos_array).T
                full_dk = [np.array(full_dk[:, i, :])
                           for i in range(self.rel_pos_array.shape[1])]
            else:
                jacobian = None

            full_kernel = [np.array(kernel(self.rel_pos_array[:, i]))
                           for i in range(self.rel_pos_array.shape[1])]
            for inf_cell_ids in self.all_inf_cell_ids:
                inf_state = copy.copy(self.initial_inf)
                for inf_cell in inf_cell_ids:
                    dot = np.dot(inf_state, full_kernel[inf_cell])
                    if primary_rate is not None:
                        dot += primary_rate
                    log_lik += np.log(dot)
                    if jac is not None:
                        if primary_rate is not None:
                            jacobian += np.append(np.dot(inf_state, full_dk[inf_cell]), 1.0) / dot
                        else:
                            jacobian += np.dot(inf_state, full_dk[inf_cell]) / dot
                    inf_state[inf_cell] += 1

            return (log_lik, jacobian)

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
                                matrix=self.matrix, primary_factor=self.primary_factor)

        elif self.precompute_level == "partial":
            np.savez_compressed(savefile, level="partial", identifier=identifier,
                                distances=self.distances, const_factors=self.const_factors,
                                all_inf_cell_ids=self.all_inf_cell_ids,
                                initial_inf=self.initial_inf, rel_pos_array=self.rel_pos_array,
                                primary_factor=self.primary_factor)

        else:
            raise ValueError("Unrecognised precomputation level!")


    def _inf_cell(self, inf_id, inf_state_m1, full_kernel):

        lik_term = theano.tensor.sum(theano.tensor.dot(inf_state_m1, full_kernel[:, inf_id]))

        new_inf_state = theano.tensor.set_subtensor(inf_state_m1[inf_id], inf_state_m1[inf_id]+1)

        return [new_inf_state, lik_term]
