"""Implementation of raster based epidemic ODE model, for use in optimal control."""

import warnings
import subprocess
import os
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import raster_tools


class RasterModel:
    """Class containing a particular SIR model parametrisation, for running and optimising control.

    Initialisation requires a dictionary of the following parameters:
        'inf_rate':         Infection Rate,
        'control_rate':     Control Rate,
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
        self._required_keys = ['inf_rate', 'control_rate', 'coupling', 'times',
                               'max_hosts', 'primary_rate']

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

    def run_scheme(self, thin_scheme=None, rogue_scheme=None, euler=False):
        """Run ODE system forward using supplied control schemes."""

        ode = integrate.ode(self.deriv)
        ode.set_integrator('vode', nsteps=1000, method='bdf')
        ode.set_initial_value(self.state_init,
                              self.params['times'][0])
        ode.set_f_params(thin_scheme, rogue_scheme)

        ts = [self.params['times'][0]]
        xs = [self.state_init]

        if euler:
            for time in self.params['times'][1:]:
                time_step = time - ts[-1]
                new_x = xs[-1] + (time_step * self.deriv(ts[-1], xs[-1], thin_scheme, rogue_scheme))
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
        u_dict = {'time': ts}
        v_dict = {'time': ts}
        results_u = None
        results_v = None

        for cell in range(self.ncells):
            s_states = [x[2*cell] for i, x in enumerate(xs)]
            s_dict['Cell' + str(cell)] = s_states

            i_states = [x[2*cell+1] for i, x in enumerate(xs)]
            i_dict['Cell' + str(cell)] = i_states

            if thin_scheme is not None:
                u_states = [thin_scheme(ts[i])[cell] for i, x in enumerate(xs)]
                u_dict['Cell' + str(cell)] = u_states

            if rogue_scheme is not None:
                v_states = [rogue_scheme(ts[i])[cell] for i, x in enumerate(xs)]
                v_dict['Cell' + str(cell)] = v_states

        results_s = pd.DataFrame(s_dict)
        results_i = pd.DataFrame(i_dict)
        if thin_scheme is not None:
            results_u = pd.DataFrame(u_dict)

        if rogue_scheme is not None:
            results_v = pd.DataFrame(v_dict)

        return RasterRun(self.params, (results_s, results_i, results_u, results_v))

    def no_control_policy(self, time, state):
        """Run policy for no control, to use with run_scheme."""
        return [0]*self.ncells

    def optimise_ipopt(self, options=None, verbose=True, method="euler", warm_start_stub=None):
        """Run optimisation using Ipopt"""

        ipopt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Ipopt")

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
            subprocess.run([os.path.join(ipopt_dir, "RasterModel.exe"), *cmd_options])
        else:
            subprocess.run([os.path.join(ipopt_dir, "RasterModel.exe"), *cmd_options],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        results_s = pd.read_csv("output_S.csv")
        results_i = pd.read_csv("output_I.csv")
        results_u = pd.read_csv("output_u.csv")
        results_v = pd.read_csv("output_v.csv")

        return RasterRun(self.params, (results_s, results_i, results_u, results_v))

    def deriv(self, t, X, thin_scheme, rogue_scheme):
        """Calculate state derivative"""

        dX = np.zeros(len(X))
        S_state = X[0::2]
        I_state = X[1::2]

        infection_terms = (self.params['primary_rate'] * S_state +
                           self.params['inf_rate']*S_state*np.dot(self.params['coupling'], I_state))

        dS = -1*infection_terms

        if thin_scheme is not None:
            thin_val = thin_scheme(t)
            dS -= np.array([
                self.params['control_rate'] * thin_val[i] for i in range(self.ncells)]) * S_state

        dI = infection_terms

        if rogue_scheme is not None:
            rogue_val = rogue_scheme(t)
            dI -= np.array([
                self.params['control_rate'] * rogue_val[i] for i in range(self.ncells)]) * I_state

        dX[0::2] = dS
        dX[1::2] = dI

        return dX


class RasterRun:
    """Class to hold results of RasterModel run(s) adn/or optimisations, for plotting/storage."""

    def __init__(self, model_params, results):
        self.model_params = model_params
        self.results_s, self.results_i, self.results_u, self.results_v = results

    def get_plot(self, time, ax_state, ax_thin=None, ax_rogue=None):
        """Generate plot of state and control at specified time."""

        # Find index in results coresponding to required time
        idx = int(self.results_s.index[self.results_s['time'] == time][0])

        data_rows = (self.results_s.iloc[idx], self.results_i.iloc[idx],
                     self.results_u.iloc[idx], self.results_v.iloc[idx])
        colours1, colours2, colours3 = self._get_colours(data_rows)

        im1 = ax_state.imshow(colours1, origin="upper")

        if ax_thin is not None:
            im2 = ax_thin.imshow(colours2, origin="upper")
        else:
            im2 = None

        if ax_rogue is not None:
            im3 = ax_rogue.imshow(colours3, origin="upper")
        else:
            im3 = None

        # Add text showing time
        _ = ax_state.text(0.02, 0.95, 'time = %.3f' % data_rows[0]['time'],
                          transform=ax_state.transAxes, weight="bold", fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.6))

        return im1, im2, im3

    def make_video(self, video_length=5):
        """Make animation of raster run."""

        # Video properties
        video_length *= 1000
        fps = 30
        nframes = fps * video_length
        interval = max([1, int(len(self.results_s) / nframes)])
        nframes = int(np.ceil(len(self.results_s)/interval)) + 1

        # Colour mappings for thinning
        cmap_thin = plt.get_cmap("Greens")
        cmap_rogue = plt.get_cmap("Oranges")
        cNorm = colors.Normalize(vmin=0, vmax=0.1)
        scalarMap_thin = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap_thin)
        scalarMap_rogue = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap_rogue)

        # Join indices into a list
        indices = np.r_[0:len(self.results_s):interval, -1]

        times = np.array(self.results_s['time'].values[indices])

        # Extract correct data points
        size = (len(times), *self.model_params['dimensions'])
        s_values = np.array([self.results_s['Cell'+str(cell)].values[indices]
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        s_values = np.reshape(s_values, size)
        i_values = np.array([self.results_i['Cell'+str(cell)].values[indices]
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        i_values = np.reshape(i_values, size)

        # Generate matrices for state colours
        colours = np.zeros((len(times), *self.model_params['dimensions'], 4))
        colours[:, :, :, 0] = i_values/self.model_params['max_hosts']
        colours[:, :, :, 1] = s_values/self.model_params['max_hosts']
        colours[:, :, :, 3] = np.ones(size)

        n_plots = 1
        if self.results_u is not None:
            n_plots += 1
            u_values = np.array([self.results_u['Cell'+str(cell)].values[indices]
                                 for cell in range(np.prod(self.model_params['dimensions']))]).T
            u_values = np.reshape(u_values, size)

            colours_thin = np.zeros((len(times), *self.model_params['dimensions'], 4))
            colours_thin = np.array([scalarMap_thin.to_rgba(u_values[i]) for i in range(size[0])])

        if self.results_v is not None:
            n_plots += 1
            v_values = np.array([self.results_v['Cell'+str(cell)].values[indices]
                                 for cell in range(np.prod(self.model_params['dimensions']))]).T
            v_values = np.reshape(v_values, size)

            colours_rogue = np.zeros((len(times), *self.model_params['dimensions'], 4))
            colours_rogue = np.array([scalarMap_rogue.to_rgba(v_values[i]) for i in range(size[0])])

        fig = plt.figure()
        ax1 = fig.add_subplot(1, n_plots, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("State")

        if self.results_u is not None:
            ax2 = fig.add_subplot(1, n_plots, 2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title("Thinning")

        if self.results_v is not None:
            ax3 = fig.add_subplot(1, n_plots, n_plots)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title("Roguing")

        fig.tight_layout()

        im1 = ax1.imshow(colours[0], animated=True, origin="upper")

        if self.results_u is not None:
            im2 = ax2.imshow(colours_thin[0], animated=True, origin="upper")
        else:
            im2 = None
        if self.results_v is not None:
            im3 = ax3.imshow(colours_rogue[0], animated=True, origin="upper")
        else:
            im3 = None
        time_text = ax1.text(0.03, 0.9, 'time = %.3f' % times[0],
                             transform=ax1.transAxes, weight="bold", fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.6))

        def update(frame_number):
            im1.set_array(colours[frame_number])
            ret_list = [im1]

            if self.results_u is not None:
                im2.set_array(colours_thin[frame_number])
                ret_list.append(im2)
            if self.results_u is not None:
                im3.set_array(colours_rogue[frame_number])
                ret_list.append(im3)

            time_text.set_text('time = %.3f' % times[frame_number])
            ret_list.append(time_text)

            return ret_list

        im_ani = animation.FuncAnimation(fig, update, interval=video_length/nframes, frames=nframes,
                                         blit=True, repeat=False)

        return im_ani

    def plot(self, video_length=5):
        """View animation of raster run."""

        _ = self.make_video(video_length)
        plt.show()

    def export_video(self, filename, video_length=5):
        """Export video as mp4"""

        im_ani = self.make_video(video_length)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=500, codec="h264")
        im_ani.save(filename+'.mp4', writer=writer, dpi=300)

    def export(self, filestub):
        """Export results to file(s)."""
        self.results_s.to_csv(filestub + "_S.csv")
        self.results_i.to_csv(filestub + "_I.csv")
        self.results_u.to_csv(filestub + "_u.csv")
        self.results_v.to_csv(filestub + "_v.csv")

    def _get_colours(self, data_rows):
        """Calculate cell colours"""

        cmap_thin = plt.get_cmap("Greens")
        cmap_rogue = plt.get_cmap("Oranges")
        cNorm = colors.Normalize(vmin=0, vmax=1.0)
        scalarMap_thin = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap_thin)
        scalarMap_rogue = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap_rogue)

        landscape_shape = (self.model_params['dimensions'][0], self.model_params['dimensions'][1])

        s_values = np.array([data_rows[0]['Cell'+str(cell)].values
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        s_values = np.reshape(s_values, landscape_shape)
        i_values = np.array([data_rows[1]['Cell'+str(cell)].values
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        i_values = np.reshape(i_values, landscape_shape)
        u_values = np.array([data_rows[2]['Cell'+str(cell)].values
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        u_values = np.reshape(u_values, landscape_shape)
        v_values = np.array([data_rows[3]['Cell'+str(cell)].values
                             for cell in range(np.prod(self.model_params['dimensions']))]).T
        v_values = np.reshape(v_values, landscape_shape)

        # Generate matrices for state and control colours
        colours1 = np.zeros((*self.model_params['dimensions'], 4))
        colours1[:, :, 0] = i_values/self.model_params['max_hosts']
        colours1[:, :, 1] = s_values/self.model_params['max_hosts']
        colours1[:, :, 3] = np.ones(landscape_shape)

        colours_thin = np.zeros((*self.model_params['dimensions'], 4))
        colours_thin = np.array(scalarMap_thin.to_rgba(u_values))

        colours_rogue = np.zeros((*self.model_params['dimensions'], 4))
        colours_rogue = np.array(scalarMap_rogue.to_rgba(v_values))

        return colours1, colours_thin, colours_rogue


class RasterOptimisation:
    """Class to hold results of Ipopt optimisations, and model setup."""

    def __init__(self, output_file_stub="output", input_file_stub=""):
        # First read setup parameters from log file.
        with open(output_file_stub + ".log", "r") as infile:
            setup_lines = infile.readlines()

        setup_dict = {}
        for line in setup_lines:
            arg, val = line.split()
            if arg != "START_FILE_STUB":
                val = float(val)

            setup_dict[arg.lower()] = val

        self.setup = setup_dict

        self.results_s = pd.read_csv(output_file_stub + "_S.csv")
        self.results_i = pd.read_csv(output_file_stub + "_I.csv")
        self.results_u = pd.read_csv(output_file_stub + "_u.csv")
        self.results_v = pd.read_csv(output_file_stub + "_v.csv")

        model_params = {}
        model_params['inf_rate'] = setup_dict['beta']
        model_params['control_rate'] = setup_dict['control_rate']
        model_params['max_hosts'] = setup_dict['max_hosts']
        model_params['primary_rate'] = 0.0

        s0_raster = raster_tools.RasterData.from_file(input_file_stub + "S0_raster.txt")
        i0_raster = raster_tools.RasterData.from_file(input_file_stub + "I0_raster.txt")
        n_raster = raster_tools.RasterData.from_file(input_file_stub + "HostDensity_raster.txt")

        dimensions = (11, 9)
        ncells = np.prod(dimensions)
        coupling = np.zeros((ncells, ncells))
        for i in range(ncells):
            for j in range(ncells):
                dx = abs((i % dimensions[0]) - (j % dimensions[0]))
                dy = abs(int(i/dimensions[0]) - int(j/dimensions[0]))
                dist = np.sqrt(dx*dx + dy*dy)
                if (dx <= 2) and (dy <= 2):
                    coupling[i, j] = np.exp(-dist/0.210619) / (2 * np.pi * 0.210619 * 0.210619)
        model_params['coupling'] = coupling

        model_params['times'] = self.results_u['time']

        self.model_params = model_params
        self.s0_raster = s0_raster.array.flatten() * n_raster.array.flatten()
        self.i0_raster = i0_raster.array.flatten() * n_raster.array.flatten()

    def run_model(self):
        """Run raster ODE model, using this optimised control."""

        model = RasterModel(self.model_params)

        thin_scheme = interp1d(self.results_u['time'], self.results_u.values[:, 1:].T,
                               kind="zero", fill_value="extrapolate")

        rogue_scheme = interp1d(self.results_v['time'], self.results_v.values[:, 1:].T,
                                kind="zero", fill_value="extrapolate")

        return model.run_scheme(thin_scheme, rogue_scheme)
