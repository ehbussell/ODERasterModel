"""Implementation of raster based epidemic ODE model, for use in optimal control."""

import warnings
import subprocess
import os
import bocop_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import colors
from matplotlib.animation import FuncAnimation


class RasterModel:
    """Class containing a particular model parameterisation, for running and optimising control.

    Initialisation requires a dictionary of the following parameters:
        'inf_rate':         Infection Rate,
        'control_rate':     Control Rate,
        'max_budget_rate':  Maximum control expenditure,
        'coupling':         Transmission kernel,
        'state_init':       Initial state [S_0,I_0...S_N-1, I_N-1],
        'N_individuals'     Number of individuals in each cell [N_0...N_N-1],
        'times':            Times to solve for,
        'dimensions':       (Nx, Ny) for dimensions of raster grid,
        'scale':            Kernel scale parameter,
     """

    def __init__(self, params):
        self.required_keys = ['inf_rate', 'control_rate', 'max_budget_rate', 'coupling',
                              'state_init', 'N_individuals', 'times', 'dimensions', 'scale']

        for key in self.required_keys:
            if key not in params:
                raise KeyError("Parameter {0} not found!".format(key))

        self.params = {k: params[k] for k in self.required_keys}

        for key in params:
            if key not in self.required_keys:
                warnings.warn("Unused parameter: {0}".format(key))

        self.ncells = np.prod(self.params['dimensions'])

    def run_scheme(self, control_scheme):
        """Run ODE system forward using supplied control scheme."""

        ode = integrate.ode(self.deriv)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(self.params['state_init'],
                              self.params['times'][0])
        ode.set_f_params(control_scheme)

        ts = [self.params['times'][0]]
        xs = [self.params['state_init']]

        for time in self.params['times'][1:]:
            ode.integrate(time)
            ts.append(ode.t)
            xs.append(ode.y)

        # Create RasterRun object to hold result
        res_dict = {'time': ts}
        for cell in range(self.ncells):
            states = [(x[2*cell], x[2*cell+1],
                       self.params['N_individuals'][cell]) for x in xs]
            res_dict['Cell' + str(cell)] = states

        results = pd.DataFrame(res_dict)

        return RasterRun(self.params, results)

    def optimise(self, BOCOP_dir=None, verbose=True):

        if BOCOP_dir is None:
            BOCOP_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BOCOP")
        set_BOCOP_params(self.params, folder=BOCOP_dir)

        if verbose is True:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")], cwd=BOCOP_dir, shell=True)
        else:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")],
                           cwd=BOCOP_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        Xt, Lt, Ut = bocop_utils.readSolFile(BOCOP_dir + "/problem.sol")

        res_dict = {'time': self.params['times']}
        for cell in range(self.ncells):
            states = [(Xt(t)[2*cell], Xt(t)[2*cell+1],
                       self.params['N_individuals'][cell]) for t in self.params['times']]
            res_dict['Cell' + str(cell)] = states

        results = pd.DataFrame(res_dict)

        return RasterRun(self.params, results)

    def deriv(self, t, X, control):
        dX = np.zeros(len(X))
        S_state = self.params['N_individuals']*X[0::2]
        I_state = self.params['N_individuals']*X[1::2]
        control_val = control(t)
        infection_terms = self.params['inf_rate']*S_state*np.dot(self.params['coupling'], I_state)

        dS = -1*infection_terms
        dI = infection_terms - np.array([
            0.0 + self.params['control_rate']*control_val[i] for i in range(self.ncells)])*I_state

        dX[0::2] = dS
        dX[1::2] = dI

        return dX


class RasterRun:
    """Class to hold results of RasterModel run(s), stored alongside model setup."""

    def __init__(self, model_params, results):
        self.model_params = model_params
        self.results = results

    def plot(self):
        """Plot RasterRun results."""

        frame_rate = 30
        animation_length = 5
        frame_interval = max(self.results['time']) / (animation_length * frame_rate)
        nframes = len(self.results)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])

        data_row = self.results.iloc[0]
        colours = self.get_colours(data_row)
        im = plt.imshow(colours, animated=True)
        time_text = ax.text(0.02, 0.95, 'time = %.3f' % data_row['time'], transform=ax.transAxes,
                            weight="bold", fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        def update(frame_number):
            data_row = self.results.iloc[frame_number]
            new_time = data_row['time']

            colours = self.get_colours(data_row)
            im.set_array(colours)
            time_text.set_text('time = %.3f' % new_time)

            return im, time_text

        animation = FuncAnimation(fig, update, interval=1000/frame_rate, frames=nframes, blit=True,
                                  repeat=False)
        plt.show()

    def export(self):
        """Export results to file(s)."""

    def get_colours(self, data_row):
        ncells = np.prod(self.model_params['dimensions'])
        colours = np.zeros((*self.model_params['dimensions'], 4))

        for i in range(ncells):
            S, I, N = data_row['Cell' + str(i)]
            x = i % self.model_params['dimensions'][0]
            y = int(i/self.model_params['dimensions'][0])
            colours[x, y, :] = (I, S, 0, N)

        return colours


def set_BOCOP_params(params, folder="BOCOP"):

    allLines = []
    # Dimensions
    allLines.append("# Dimensions\n")
    ncells = np.prod(params['dimensions'])
    dim_string = str(2*ncells) + " " + str(2*ncells) + " " + str(ncells) + " 0 0 1\n"
    allLines.append(dim_string)

    # Initial conditions
    allLines.append("# Initial Conditions\n")
    for cond in params['state_init']:
        init_string = str(cond) + " " + str(cond) + " equal\n"
        allLines.append(init_string)

    # State bounds
    allLines.append("# State Bounds\n")
    for i in range(2*ncells):
        allLines.append(">" + str(i) + ":1:" + str(i) + " 0 2e+020 lower\n")

    # Control bounds
    allLines.append("# Control Bounds\n")
    for i in range(ncells):
        allLines.append("0 1 both\n")

    # Path consraint bounds
    allLines.append("# Path Constraint Bounds\n")
    allLines.append("-2e+020 " + str(params['max_budget_rate']) + " upper\n")

    with open(folder + "/problem.bounds", "w") as f:
        f.writelines(allLines)

    with open(folder + "/problem.constants", "r") as f:
        allLines = f.readlines()

    allLines[5] = str(params['inf_rate']) + "\n"
    allLines[6] = str(params['control_rate']) + "\n"
    allLines[7] = str(params['dimensions'][0]) + "\n"
    allLines[8] = str(params['dimensions'][1]) + "\n"
    allLines[9] = str(params['scale']) + "\n"

    with open(folder + "/problem.constants", "w") as f:
        f.writelines(allLines)

    with open(folder + "/problem.def", "r") as f:
        allLines = f.readlines()

    nSteps = str(len(params['times']) - 1)
    allLines[5] = "time.initial double " + str(params['times'][0]) + "\n"
    allLines[6] = "time.final double " + str(params['times'][-1]) + "\n"
    allLines[18] = "discretization.steps integer " + nSteps + "\n"

    allLines[9] = "state.dimension integer " + str(2*ncells) + "\n"
    allLines[10] = "control.dimension integer " + str(ncells) + "\n"
    allLines[14] = "boundarycond.dimension integer " + str(2*ncells) + "\n"

    with open(folder + "/problem.def", "w") as f:
        f.writelines(allLines)
