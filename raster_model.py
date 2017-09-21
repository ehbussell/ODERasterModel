"""Implementation of raster based epidemic ODE model, for use in optimal control."""

import warnings
import subprocess
import os
import inspect
import bocop_utils
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
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

        # print(inspect.stack()[1:])

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
                       self.params['N_individuals'][cell], control_scheme(ts[i])[cell])
                      for i, x in enumerate(xs)]
            res_dict['Cell' + str(cell)] = states

        results = pd.DataFrame(res_dict)

        return RasterRun(self.params, results)

    def optimise(self, BOCOP_dir=None, verbose=True):

        if BOCOP_dir is None:
            BOCOP_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BOCOP")
        set_BOCOP_params(self.params, folder=BOCOP_dir)

        if verbose is True:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")], cwd=BOCOP_dir)
        else:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")],
                           cwd=BOCOP_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        Xt, Lt, Ut = bocop_utils.readSolFile(BOCOP_dir + "/problem.sol")

        res_dict = {'time': self.params['times']}
        for cell in range(self.ncells):
            states = [(Xt(t)[2*cell], Xt(t)[2*cell+1],
                       self.params['N_individuals'][cell],
                       Ut(t)[cell]) for t in self.params['times']]
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

        video_length = 5
        frame_rate = 30
        nframes = len(self.results)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.tight_layout()

        data_row = self.results.iloc[0]
        colours1, colours2 = self.get_colours(data_row)
        im1 = ax1.imshow(colours1, animated=True)
        im2 = ax2.imshow(colours2, animated=True)
        time_text = ax1.text(0.02, 0.95, 'time = %.3f' % data_row['time'], transform=ax1.transAxes,
                             weight="bold", fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

        def update(frame_number):
            data_row = self.results.iloc[frame_number]
            new_time = data_row['time']

            colours1, colours2 = self.get_colours(data_row)
            im1.set_array(colours1)
            im2.set_array(colours2)
            time_text.set_text('time = %.3f' % new_time)

            return im1, im2, time_text

        animation = FuncAnimation(fig, update, interval=video_length/nframes, frames=nframes,
                                  blit=True, repeat=False)
        plt.show()

    def export(self, filestub):
        """Export results to file(s)."""
        self.results.to_csv(filestub)

    def get_colours(self, data_row):
        cmap = plt.get_cmap("Oranges")
        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        ncells = np.prod(self.model_params['dimensions'])
        colours1 = np.zeros((*self.model_params['dimensions'], 4))
        colours2 = np.zeros((*self.model_params['dimensions'], 4))

        for i in range(ncells):
            S, I, N, f = data_row['Cell' + str(i)]
            x = i % self.model_params['dimensions'][0]
            y = int(i/self.model_params['dimensions'][0])
            colours1[x, y, :] = (I, S, 0, N)
            colours2[x, y, :] = scalarMap.to_rgba(f)

        return colours1, colours2


def set_BOCOP_params(params, folder="BOCOP"):

    all_lines = []
    # Dimensions
    all_lines.append("# Dimensions\n")
    ncells = np.prod(params['dimensions'])
    dim_string = str(2*ncells) + " " + str(2*ncells) + " " + str(ncells) + " 0 0 1\n"
    all_lines.append(dim_string)

    # Initial conditions
    all_lines.append("# Initial Conditions\n")
    for cond in params['state_init']:
        init_string = str(cond) + " " + str(cond) + " equal\n"
        all_lines.append(init_string)

    # State bounds
    all_lines.append("# State Bounds\n")
    for i in range(2*ncells):
        all_lines.append(">" + str(i) + ":1:" + str(i) + " 0 2e+020 lower\n")

    # Control bounds
    all_lines.append("# Control Bounds\n")
    for i in range(ncells):
        all_lines.append("0 1 both\n")

    # Path consraint bounds
    all_lines.append("# Path Constraint Bounds\n")
    all_lines.append("-2e+020 " + str(params['max_budget_rate']) + " upper\n")

    with open(folder + "/problem.bounds", "w") as f:
        f.writelines(all_lines)

    with open(folder + "/problem.constants", "r") as f:
        all_lines = f.readlines()

    all_lines[5] = str(params['inf_rate']) + "\n"
    all_lines[6] = str(params['control_rate']) + "\n"
    all_lines[7] = str(params['dimensions'][0]) + "\n"
    all_lines[8] = str(params['dimensions'][1]) + "\n"
    all_lines[9] = str(params['scale']) + "\n"

    with open(folder + "/problem.constants", "w") as f:
        f.writelines(all_lines)

    with open(folder + "/problem.def", "r") as f:
        all_lines = f.readlines()

    nsteps = str(len(params['times']) - 1)
    all_lines[5] = "time.initial double " + str(params['times'][0]) + "\n"
    all_lines[6] = "time.final double " + str(params['times'][-1]) + "\n"
    all_lines[18] = "discretization.steps integer " + nsteps + "\n"

    all_lines[9] = "state.dimension integer " + str(2*ncells) + "\n"
    all_lines[10] = "control.dimension integer " + str(ncells) + "\n"
    all_lines[14] = "boundarycond.dimension integer " + str(2*ncells) + "\n"

    with open(folder + "/problem.def", "w") as f:
        f.writelines(all_lines)
