"""Implementation of raster based epidemic ODE model, for use in optimal control."""

import warnings
import subprocess
import pdb
import os
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import bocop_utils
import raster_tools


class RasterModel:
    """Class containing a particular SIR model parametrisation, for running and optimising control.

    Initialisation requires a dictionary of the following parameters:
        'inf_rate':         Infection Rate,
        'control_rate':     Control Rate,
        'max_budget_rate':  Maximum control expenditure,
        'coupling':         Transmission kernel,
        'times':            Times to solve for,
        'max_hosts':        Maximum number of host units per cell,
        'primary_rate':     Primary infection rate
    And optional arguments:
        host_density_file   Name of ESRI ASCII raster file containing host density,
                            default: "HostDensity_raster.txt"
        initial_s_file      Name of ESRI ASCII raster file containing initial proportions in S,
                            default: "S0_raster.txt"
        initial_i_file      Name of ESRI ASCII raster file containing initial proportions in I,
                            default: "I0_raster.txt"
     """

    def __init__(self, params, host_density_file="HostDensity_raster.txt",
                 initial_s_file="S0_raster.txt", initial_i_file="I0_raster.txt"):
        self._required_keys = ['inf_rate', 'control_rate', 'max_budget_rate',
                               'coupling', 'times', 'max_hosts', 'primary_rate']

        for key in self._required_keys:
            if key not in params:
                raise KeyError("Parameter {0} not found!".format(key))

        self.params = {k: params[k] for k in self._required_keys}

        for key in params:
            if key not in self._required_keys:
                warnings.warn("Unused parameter: {0}".format(key))

        self.state_init = self._read_rasters(host_density_file, initial_s_file, initial_i_file)

        self.ncells = np.prod(self.params['dimensions'])

    def _read_rasters(self, host_density_file, initial_s_file, initial_i_file):
        """Read initialisation rasters to set initial state and dimensions."""

        host_raster = raster_tools.RasterData.from_file(host_density_file)
        s0_raster = raster_tools.RasterData.from_file(initial_s_file)
        i0_raster = raster_tools.RasterData.from_file(initial_i_file)

        assert host_raster.header_vals == s0_raster.header_vals == i0_raster.header_vals

        self.params['dimensions'] = (host_raster.header_vals['nrows'],
                                     host_raster.header_vals['ncols'])

        self._host_density = host_raster.array.flatten()

        s_state_init = np.clip(host_raster.array.flatten() * s0_raster.array.flatten() *
                               self.params['max_hosts'], 0, None)
        i_state_init = np.clip(host_raster.array.flatten() * i0_raster.array.flatten() *
                               self.params['max_hosts'], 0, None)

        state_init = np.empty((s_state_init.size + i_state_init.size), dtype=s_state_init.dtype)
        state_init[0::2] = s_state_init
        state_init[1::2] = i_state_init

        return state_init

    def set_init_state(self, host_density_file, initial_s_file, initial_i_file):
        """Initialise start point for raster run from host raster files."""

        self.state_init = self._read_rasters(host_density_file, initial_s_file, initial_i_file)

    def run_scheme(self, control_scheme, euler=False):
        """Run ODE system forward using supplied control scheme."""

        ode = integrate.ode(self.deriv)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(self.state_init,
                              self.params['times'][0])
        ode.set_f_params(control_scheme)

        ts = [self.params['times'][0]]
        xs = [self.state_init]

        if euler:
            for time in self.params['times'][1:]:
                time_step = time - ts[-1]
                new_x = xs[-1] + (time_step * self.deriv(ts[-1], xs[-1], control_scheme))
                ts.append(time)
                xs.append(new_x)
        else:
            for time in self.params['times'][1:]:
                ode.integrate(time)
                ts.append(ode.t)
                xs.append(np.clip(ode.y, 0, None))

        # Create RasterRun object to hold result
        s_dict = {'time': ts}
        i_dict = {'time': ts}
        f_dict = {'time': ts}

        for cell in range(self.ncells):
            s_states = [x[2*cell] for i, x in enumerate(xs)]
            i_states = [x[2*cell+1] for i, x in enumerate(xs)]
            f_states = [control_scheme(ts[i], xs[i])[cell] for i, x in enumerate(xs)]
            s_dict['Cell' + str(cell)] = s_states
            i_dict['Cell' + str(cell)] = i_states
            f_dict['Cell' + str(cell)] = f_states

        results_s = pd.DataFrame(s_dict)
        results_i = pd.DataFrame(i_dict)
        results_f = pd.DataFrame(f_dict)

        return RasterRun(self.params, (results_s, results_i, results_f))

    def no_control_policy(self, time, state):
        """Run policy for no control, to use with run_scheme."""
        return [0]*self.ncells

    def optimise_BOCOP(self, BOCOP_dir=None, verbose=True):
        """Run BOCOP optimisation of model"""

        if BOCOP_dir is None:
            BOCOP_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BOCOP")
        set_BOCOP_params(self.params, self.state_init, folder=BOCOP_dir)

        if verbose is True:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")], cwd=BOCOP_dir)
        else:
            subprocess.run([os.path.join(BOCOP_dir, "bocop.exe")],
                           cwd=BOCOP_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        Xt, Lt, Ut = bocop_utils.readSolFile(BOCOP_dir + "/problem.sol")

        s_dict = {'time': self.params['times']}
        i_dict = {'time': self.params['times']}
        f_dict = {'time': self.params['times']}

        for cell in range(self.ncells):
            s_states = [Xt(t)[2*cell] for t in self.params['times']]
            i_states = [Xt(t)[2*cell+1] for t in self.params['times']]
            f_states = [Ut(t)[cell] for t in self.params['times']]
            s_dict['Cell' + str(cell)] = s_states
            i_dict['Cell' + str(cell)] = i_states
            f_dict['Cell' + str(cell)] = f_states

        results_s = pd.DataFrame(s_dict)
        results_i = pd.DataFrame(i_dict)
        results_f = pd.DataFrame(f_dict)

        return RasterRun(self.params, (results_s, results_i, results_f))

    def optimise_Ipopt(self, options=None, verbose=True, method="euler", warm_start_stub=None):
        """Run optimisation using Ipopt"""

        Ipopt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Ipopt")

        if method == "euler":
            cmd_options = [
                str(0),
                str(self.params['inf_rate']),
                str(self.params['control_rate']),
                str(self.params['max_budget_rate']),
                str(self.params['times'][-1]),
                str(len(self.params['times'][:-1])),
                str(self.params['max_hosts'])
            ]
        elif method == "midpoint":
            cmd_options = [
                str(1),
                str(self.params['inf_rate']),
                str(self.params['control_rate']),
                str(self.params['max_budget_rate']),
                str(self.params['times'][-1]),
                str(len(self.params['times'][:-1])),
                str(self.params['max_hosts'])
            ]
        else:
            raise ValueError("Unknown Method!")

        if warm_start_stub is not None:
            cmd_options.append(warm_start_stub)

        if options is not None:
            with open("ipopt.opt", "w") as outfile:
                for key, value in options.items():
                    outfile.write(str(key) + " " + str(value) + "\n")

        if verbose is True:
            subprocess.run([os.path.join(Ipopt_dir, "RasterModel.exe"), *cmd_options])
        else:
            subprocess.run([os.path.join(Ipopt_dir, "RasterModel.exe"), *cmd_options],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        results_s = pd.read_csv("output_S.csv")
        results_i = pd.read_csv("output_I.csv")
        results_f = pd.read_csv("output_f.csv")

        return RasterRun(self.params, (results_s, results_i, results_f))

    def deriv(self, t, X, control):
        """Calculate state derivative"""

        dX = np.zeros(len(X))
        S_state = X[0::2]
        I_state = X[1::2]
        control_val = control(t, X)
        infection_terms = (self.params['primary_rate'] * S_state +
                           self.params['inf_rate']*S_state*np.dot(self.params['coupling'], I_state))

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
        self.results_s, self.results_i, self.results_f = results

        val_cols = [c for c in self.results_i.columns if c.startswith('Cell')]
        # print(self.results_i[val_cols].max().max())
        self.max_budget_used = self.results_i[val_cols].max().max()

    def plot_budget(self):
        """Plot budget expenditure over time."""
        # TODO implement budget plot
        pass

    def get_plot(self, time, ax_state, ax_control):
        idx = int(self.results_s.index[self.results_s['time'] == time][0])
        print(idx)
        data_rows = (self.results_s.iloc[idx], self.results_i.iloc[idx], self.results_f.iloc[idx]*self.results_i.iloc[idx])
        colours1, colours2 = self._get_colours(data_rows)
        im1 = ax_state.imshow(colours1, origin="upper")
        im2 = ax_control.imshow(colours2, origin="upper")
        time_text = ax_state.text(0.02, 0.95, 'time = %.3f' % data_rows[0]['time'],
                             transform=ax_state.transAxes, weight="bold", fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.6))

    def make_video(self, video_length=5):
        """Make animation of raster run."""

        video_length *= 1000
        fps = 30
        nframes = fps * video_length
        interval = max([1, int(len(self.results_s) / nframes)])
        nframes = int(np.ceil(len(self.results_s)/interval)) + 1

        cmap = plt.get_cmap("Oranges")
        cNorm = colors.Normalize(vmin=0, vmax=self.max_budget_used)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        indices = np.r_[0:len(self.results_s):interval, -1]

        times = np.array(self.results_s['time'].values[indices])

        size = (len(times), *self.model_params['dimensions'])
        s_values = np.array([self.results_s['Cell'+str(cell)].values[indices]
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        s_values = np.reshape(s_values, size)
        i_values = np.array([self.results_i['Cell'+str(cell)].values[indices]
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        i_values = np.reshape(i_values, size)
        f_values = np.array([self.results_f['Cell'+str(cell)].values[indices]*
                             self.results_i['Cell'+str(cell)].values[indices]
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        f_values = np.reshape(f_values, size)

        colours = np.zeros((len(times), *self.model_params['dimensions'], 4))
        colours[:, :, :, 0] = i_values/self.model_params['max_hosts']
        colours[:, :, :, 1] = s_values/self.model_params['max_hosts']
        colours[:, :, :, 3] = np.ones(size)

        colours_control = np.zeros((len(times), *self.model_params['dimensions'], 4))
        colours_control = np.array([scalarMap.to_rgba(f_values[i]) for i in range(size[0])])

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.tight_layout()

        im1 = ax1.imshow(colours[0], animated=True, origin="upper")
        im2 = ax2.imshow(colours_control[0], animated=True, origin="upper")
        time_text = ax1.text(0.03, 0.9, 'time = %.3f' % times[0],
                             transform=ax1.transAxes, weight="bold", fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.6))

        def update(frame_number):
            im1.set_array(colours[frame_number])
            im2.set_array(colours_control[frame_number])
            time_text.set_text('time = %.3f' % times[frame_number])

            return im1, im2, time_text

        im_ani = animation.FuncAnimation(fig, update, interval=video_length/nframes, frames=nframes,
                                         blit=True, repeat=False)

        return im_ani

    def plot(self, video_length=5):
        """View animation of raster run."""

        im_ani = self.make_video(video_length)

        plt.show()

    def export_video(self, filename, video_length=5):
        """Export video as html"""

        im_ani = self.make_video(video_length)

        plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Elliott\\Documents\\ffmpeg-20171107-ce52e43-win64-static\\bin\\ffmpeg.exe'
        Writer = animation.writers['html']
        writer = Writer(fps=30, bitrate=500)
        im_ani.save(filename+".html", writer=writer)

    def export(self, filestub):
        """Export results to file(s)."""
        self.results_s.to_csv(filestub + "_S.csv")
        self.results_i.to_csv(filestub + "_I.csv")
        self.results_f.to_csv(filestub + "_f.csv")

    def _get_colours(self, data_rows):
        """Calculate cell colours"""
        cmap = plt.get_cmap("Oranges")
        cNorm = colors.Normalize(vmin=0, vmax=self.max_budget_used)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        ncells = np.prod(self.model_params['dimensions'])
        colours1 = np.zeros((self.model_params['dimensions'][0], self.model_params['dimensions'][1], 4))
        colours2 = np.zeros((self.model_params['dimensions'][0], self.model_params['dimensions'][1], 4))

        for i in range(ncells):
            S = data_rows[0]['Cell' + str(i)]
            I = data_rows[1]['Cell' + str(i)]
            B = data_rows[2]['Cell' + str(i)]
            col = i % self.model_params['dimensions'][1]
            row = int(i/self.model_params['dimensions'][1])
            colours1[row, col, :] = (I/self.model_params['max_hosts'], S/self.model_params['max_hosts'], 0, 1)
            # if I > 0:
            #     colours1[col, row, :] = (I/self.model_params['max_hosts'], 0, 0, 1)
            # else:
            #     colours1[col, row, :] = (0, S/self.model_params['max_hosts'], 0, 1)

            colours2[row, col, :] = scalarMap.to_rgba(B)

        return colours1, colours2


def set_BOCOP_params(params, state_init, folder="BOCOP"):
    """Set up BOCOP initialisation files, ready for optimiation."""

    all_lines = []
    # Dimensions
    all_lines.append("# Dimensions\n")
    ncells = np.prod(params['dimensions'])
    dim_string = str(2*ncells) + " " + str(2*ncells) + " " + str(ncells) + " 0 0 1\n"
    all_lines.append(dim_string)

    # Initial conditions
    all_lines.append("# Initial Conditions\n")
    for cond in state_init:
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
