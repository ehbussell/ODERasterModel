import unittest
import os
import glob
import numpy as np
import raster_tools
import IndividualSimulator
import raster_model_fitting as rmf

class NonSpatialTests(unittest.TestCase):
    """Test fitting of raster model in non-spatial context."""

    @classmethod
    def setUpClass(cls):
        # Setup and run simulation data, multiple hosts in single cell to make non-spatial

        cls._data_stub = os.path.join("testing", "nonspatial_sim_output")
        cls._beta_val = 0.0007

        # Create host file
        host_raster = raster_tools.RasterData((1, 1), array=[[1000]])
        host_file = os.path.join("testing", "nonspatial_host_test_case.txt")
        host_raster.to_file(host_file)

        # Create initial conditions files
        init_stub = os.path.join("testing", "nonspatial_init_test_case")
        host_raster.array[0, 0] = 995
        host_raster.to_file(init_stub + "_S.txt")
        host_raster.array[0, 0] = 5
        host_raster.to_file(init_stub + "_I.txt")

        # Create kernel file
        host_raster.array[0, 0] = 1
        host_raster.to_file(init_stub + "_kernel.txt")

        # Setup config file
        config_filename = os.path.join("testing", "nonspatial_config.ini")
        config_str = "\n[Epidemiology]\n"
        config_str += "Model = SI\nInfRate = " + str(cls._beta_val)
        config_str += "\nIAdvRate = 0.0\nKernelType = RASTER\n"
        config_str += "\n[Simulation]\n"
        config_str += "SimulationType = RASTER\nFinalTime = 10.0\nNIterations = 100\n"
        config_str += "HostPosFile = " + host_file + "\nInitCondFile = " + init_stub + "\n"
        config_str += "KernelFile = " + init_stub + "_kernel.txt" + "\n"
        config_str += "VirtualSporulationStart = 1\nMaxHosts = 1"
        config_str += "\n[Output]\n"
        config_str += "SummaryOutputFreq = 0\nOutputFileStub = " + cls._data_stub
        config_str += "\n[Optimisation]\n"
        config_str += "SaveSetup = False\nRateStructure-Infection = ratetree"
        with open(config_filename, "w") as outfile:
            outfile.write(config_str)

        cls._raster_header = host_raster.header_vals

        # Run simulations
        IndividualSimulator.main(configFile=config_filename, silent=True)

    def _nonspatial_generator(self, beta):
        def nonspatial_kernel(dist):
            return beta*np.ones_like(dist)
        return nonspatial_kernel

    def _nonspatial_jac_generator(self, beta):
        def nonspatial_jac(dist):
            return np.array([np.ones_like(dist)])
        return nonspatial_jac

    def test_non_spatial_full(self):
        """Test full likelihood precalculation and MLE fitting for non-spatial test case."""
        # Fully precalculate likelihood function for non-spatial kernel (i.e. beta)
        lik_func = rmf.precompute_loglik(data_stub=self._data_stub, nsims=None,
                                         raster_header=self._raster_header, precompute_level="full")

        # Fit using MLE
        fitted_params = rmf.fit_raster_MLE(
            self._nonspatial_generator, {"beta": [0, 0.1]}, likelihood_func=lik_func,
            kernel_jac=self._nonspatial_jac_generator)

        # Assert beta value is close to that used in simulation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.01)

        return fitted_params["beta"]

    def test_non_spatial_partial(self):
        """Test partial likelihood precalculation and MLE fitting for non-spatial test case."""
        # Fully precalculate likelihood function for non-spatial kernel (i.e. beta)
        lik_func = rmf.precompute_loglik(
            data_stub=self._data_stub, nsims=None, raster_header=self._raster_header,
            precompute_level="partial")

        # Fit using MLE
        fitted_params = rmf.fit_raster_MLE(
            self._nonspatial_generator, {"beta": [0, 0.1]}, likelihood_func=lik_func,
            kernel_jac=self._nonspatial_jac_generator)

        # Assert beta value is close to that used in simulation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.01)

        return fitted_params["beta"]

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join("testing", "nonspatial_config.ini"))
        os.remove(os.path.join("testing", "nonspatial_host_test_case.txt"))
        for file in glob.glob(cls._data_stub + "*"):
            os.remove(file)
        for file in glob.glob(os.path.join("testing", "nonspatial_init_test_case_*")):
            os.remove(file)


class SpatialTests(unittest.TestCase):
    """Test fitting of raster model in spatial context."""

    @classmethod
    def setUpClass(cls):
        # Setup and run simulation data, single hosts on lattice with exponential kernel

        cls._data_stub = os.path.join("testing", "spatial_sim_output")
        cls._beta_val = 10
        cls._scale_val = 0.3
        cls._end_time = 10
        size = (20, 20)

        # Create host file
        host_raster = raster_tools.RasterData(size, array=np.ones(size))
        host_file = os.path.join("testing", "spatial_host_test_case.txt")
        host_raster.to_file(host_file)

        # Create initial conditions files
        init_stub = os.path.join("testing", "spatial_init_test_case")
        host_raster.array[int(size[0]/2), int(size[1]/2)] = 0
        host_raster.to_file(init_stub + "_S.txt")
        host_raster.array = np.zeros(size)
        host_raster.array[int(size[0]/2), int(size[1]/2)] = 1
        host_raster.to_file(init_stub + "_I.txt")

        # Create kernel file
        kernel_range = 20
        kernel_size = (2*kernel_range+1, 2*kernel_range+1)
        kernel_raster = raster_tools.RasterData(kernel_size, array=np.zeros(kernel_size))
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                row = i - kernel_range
                col = j - kernel_range
                distance = np.sqrt(row*row + col*col)
                # kernel_raster.array[i, j] = np.exp(-distance/cls._scale_val) / (
                #     2 * np.pi * cls._scale_val * cls._scale_val)
                kernel_raster.array[i, j] = np.exp(-distance/cls._scale_val)
        kernel_raster.to_file(init_stub + "_kernel.txt")

        # Setup config file
        config_filename = os.path.join("testing", "spatial_config.ini")
        config_str = "\n[Epidemiology]\n"
        config_str += "Model = SI\nInfRate = " + str(cls._beta_val) 
        config_str += "\nIAdvRate = 0.0\nKernelType = RASTER\n"
        config_str += "\n[Simulation]\n"
        config_str += "SimulationType = RASTER\nFinalTime = " + str(cls._end_time) +"\n"
        config_str += "NIterations = 100\n"
        config_str += "HostPosFile = " + host_file + "\nInitCondFile = " + init_stub + "\n"
        config_str += "KernelFile = " + init_stub + "_kernel.txt" + "\n"
        config_str += "VirtualSporulationStart = 3\nMaxHosts = 1"
        config_str += "\n[Output]\n"
        config_str += "SummaryOutputFreq = 0\nOutputFileStub = " + cls._data_stub
        config_str += "\n[Optimisation]\n"
        config_str += "SaveSetup = False\nRateStructure-Infection = ratetree"
        with open(config_filename, "w") as outfile:
            outfile.write(config_str)

        cls._raster_header = host_raster.header_vals

        # Run simulations
        IndividualSimulator.main(configFile=config_filename)

    def _spatial_generator(self, beta, scale):
        def spatial_kernel(dist):
            # return beta*np.exp(-dist / scale) / (2 * np.pi * scale * scale)
            return beta*np.exp(-dist / scale)
        return spatial_kernel

    def _spatial_jac_generator(self, beta, scale):
        def spatial_jac(dist):
            # jac = np.array([
            #     np.exp(-dist / scale) / (2 * np.pi * scale * scale),
            #     beta * ((dist/(2*scale)) - 1) * np.exp(-dist / scale) / (
            #         np.pi * scale * scale * scale)
            # ])
            jac = np.array([
                np.exp(-dist / scale),
                beta * (dist/(scale*scale)) * np.exp(-dist / scale)
            ])
            return jac
        return spatial_jac

    def test_spatial_full(self):
        """Test full likelihood precalculation and MLE fitting for spatial test case."""
        # Fully precalculate likelihood function for exponential kernel
        lik_func = rmf.precompute_loglik(data_stub=self._data_stub, nsims=None,
                                         raster_header=self._raster_header, precompute_level="full",
                                         end_time=self._end_time)

        # Fit using MLE
        fitted_params, raw_output = rmf.fit_raster_MLE(
            self._spatial_generator, {"beta": [0, 20], "scale": [0, 10]}, likelihood_func=lik_func,
            kernel_jac=self._spatial_jac_generator, raw_output=True,
            param_start={'beta':8, 'scale':0.2})

        print(fitted_params, raw_output)

        # Assert beta and scale values are close to those used in simulation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.05)
        self.assertTrue(abs((self._scale_val-fitted_params['scale'])/self._scale_val) < 0.05)
        return fitted_params['beta'], fitted_params['scale']

    def test_spatial_partial(self):
        """Test partial likelihood precalculation and MLE fitting for spatial test case."""
        # Fully precalculate likelihood function for exponential kernel
        lik_func = rmf.precompute_loglik(
            data_stub=self._data_stub, nsims=None, raster_header=self._raster_header,
            precompute_level="partial", end_time=self._end_time)

        # Fit using MLE
        fitted_params, raw_output = rmf.fit_raster_MLE(
            self._spatial_generator, {"beta": [0, 20], "scale": [0, 10]}, likelihood_func=lik_func,
            kernel_jac=self._spatial_jac_generator, raw_output=True,
            param_start={'beta':8, 'scale':0.2})

        print(fitted_params, raw_output)

        # Assert beta and scale values are close to those used in simulation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.05)
        self.assertTrue(abs((self._scale_val-fitted_params['scale'])/self._scale_val) < 0.05)
        return fitted_params['beta'], fitted_params['scale']

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join("testing", "spatial_config.ini"))
        os.remove(os.path.join("testing", "spatial_host_test_case.txt"))
        for file in glob.glob(cls._data_stub + "*"):
            os.remove(file)
        for file in glob.glob(os.path.join("testing", "spatial_init_test_case_*")):
            os.remove(file)


class TargetRasterTests(unittest.TestCase):
    """Test fitting functions when only a subset of data is used for fitting."""

    @classmethod
    def setUpClass(cls):
        # Setup and run simulation data, single hosts on lattice with exponential kernel

        cls._data_stub = os.path.join("testing", "target_sim_output")
        cls._beta_val = 6
        cls._scale_val = 0.9
        cls._end_time = 10.0
        cls._size = (21, 21)

        # Create host file
        motif = np.zeros((3, 3))
        motif[1, 1] = 1
        host_array = np.tile(motif, cls._size)
        # host_array = host_array * np.random.randint(1, 4, host_array.shape)
        host_raster = raster_tools.RasterData([3*x for x in cls._size], array=host_array)
        host_file = os.path.join("testing", "target_host_test_case.txt")
        host_raster.to_file(host_file)

        # Create initial conditions files
        centre_host = tuple(3*int(x/2) + 1 for x in cls._size)
        init_stub = os.path.join("testing", "target_init_test_case")
        n_init_inf = host_raster.array[centre_host]
        host_raster.array[centre_host] = 0
        host_raster.to_file(init_stub + "_S.txt")
        host_raster.array = np.zeros_like(host_array)
        host_raster.array[centre_host] = n_init_inf
        host_raster.to_file(init_stub + "_I.txt")

        # TODO think about kernel normalisation

        # Create kernel file
        kernel_range = max(cls._size)*3
        kernel_size = (2*kernel_range+1, 2*kernel_range+1)
        kernel_raster = raster_tools.RasterData(kernel_size, array=np.zeros(kernel_size))
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                row = i - kernel_range
                col = j - kernel_range
                distance = np.sqrt(row*row + col*col)
                # host_raster.array[i, j] = np.exp(-distance/cls._scale_val) / (
                #     2 * np.pi * cls._scale_val * cls._scale_val)
                if row % 3 == 0 and col % 3 == 0:
                    kernel_raster.array[i, j] = np.exp(-distance/cls._scale_val)

        kernel_raster.to_file(init_stub + "_kernel.txt")

        # Setup config file
        config_filename = os.path.join("testing", "target_config.ini")
        config_str = "\n[Epidemiology]\n"
        config_str += "Model = SI\nInfRate = " + str(cls._beta_val)
        config_str += "\nIAdvRate = 0.0\nKernelType = RASTER\n"
        config_str += "\n[Simulation]\n"
        config_str += "SimulationType = RASTER\nFinalTime = " + str(cls._end_time) + "\n"
        config_str += "NIterations = 200\n"
        config_str += "HostPosFile = " + host_file + "\nInitCondFile = " + init_stub + "\n"
        config_str += "KernelFile = " + init_stub + "_kernel.txt" + "\n"
        config_str += "VirtualSporulationStart = 3\nMaxHosts = 1"
        config_str += "\n[Output]\n"
        config_str += "SummaryOutputFreq = 0\nOutputFileStub = " + cls._data_stub
        config_str += "\n[Optimisation]\n"
        config_str += "SaveSetup = False\nRateStructure-Infection = ratetree"
        with open(config_filename, "w") as outfile:
            outfile.write(config_str)

        # Run simulations
        IndividualSimulator.main(configFile=config_filename)

    def _spatial_generator(self, beta, scale):
        def spatial_kernel(dist):
            # return beta*np.exp(-dist / scale) / (2 * np.pi * scale * scale)
            return beta*np.exp(-dist / scale)
        return spatial_kernel

    def _spatial_jac_generator(self, beta, scale):
        def spatial_jac(dist):
            # jac = np.array([
            #     np.exp(-dist / scale) / (2 * np.pi * scale * scale),
            #     beta * ((dist/(2*scale)) - 1) * np.exp(-dist / scale) / (
            #         np.pi * scale * scale * scale)
            # ])
            jac = np.array([
                np.exp(-dist / scale),
                beta * (dist/(scale*scale)) * np.exp(-dist / scale)
            ])
            return jac
        return spatial_jac

    def test_subset_fitting(self):
        """Test raster fitting on subset of simulation output"""
        centre = [3*int(x / 2)+1 for x in self._size]
        new_size = [3*(int(x / 2) + 1) for x in self._size]
        start = [int(cen - (n-1)/2) for cen, n in zip(centre, new_size)]

        raster_header = {
            'nrows': new_size[0],
            'ncols': new_size[1],
            'xllcorner': start[0],
            'yllcorner': start[1],
            'cellsize': 1,
            'NODATA_value': -9999
        }

        print(raster_header)

        # Fully precalculate likelihood function for exponential kernel
        lik_func = rmf.precompute_loglik(data_stub=self._data_stub, nsims=None,
                                         raster_header=raster_header, precompute_level="full",
                                         ignore_outside_raster=True)

        # Fit using MLE
        fitted_params, raw_output = rmf.fit_raster_MLE(
            self._spatial_generator, {"beta": [0, 50], "scale": [0, 10]}, likelihood_func=lik_func,
            kernel_jac=self._spatial_jac_generator, raw_output=True,
            param_start={'beta':4, 'scale':0.2})

        print(fitted_params, raw_output)

        # Assert beta and scale values are close to those used in simulation
        # Specifically allow 10% for beta since data subsetting will result in overestimation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.1)
        self.assertTrue(abs((self._scale_val-fitted_params['scale'])/self._scale_val) < 0.05)

    def test_aggregate_fitting(self):
        """Test fitting when simulation hosts aggregated to reduced resolution."""
        centre = [3*int(x / 2)+1 for x in self._size]
        new_size = [(int(x / 2) + 1) for x in self._size]
        start = [int(cen - (3*n-1)/2) for cen, n in zip(centre, new_size)]

        raster_header = {
            'nrows': new_size[0],
            'ncols': new_size[1],
            'xllcorner': start[0],
            'yllcorner': start[1],
            'cellsize': 3,
            'NODATA_value': -9999
        }

        print(raster_header)

        # Fully precalculate likelihood function for exponential kernel
        lik_func = rmf.precompute_loglik(data_stub=self._data_stub, nsims=None,
                                         raster_header=raster_header, precompute_level="full",
                                         ignore_outside_raster=True)

        # Fit using MLE
        fitted_params, raw_output = rmf.fit_raster_MLE(
            self._spatial_generator, {"beta": [0, 50], "scale": [0, 10]}, likelihood_func=lik_func,
            kernel_jac=self._spatial_jac_generator, raw_output=True,
            param_start={'beta':4, 'scale':0.2})

        print(fitted_params, raw_output)

        # Assert beta and scale values are close to those used in simulation
        # Specifically allow 10% for beta since data subsetting will result in overestimation
        self.assertTrue(abs((self._beta_val-fitted_params['beta'])/self._beta_val) < 0.1)
        self.assertTrue(abs((self._scale_val-3*fitted_params['scale'])/self._scale_val) < 0.05)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join("testing", "target_config.ini"))
        os.remove(os.path.join("testing", "target_host_test_case.txt"))
        for file in glob.glob(cls._data_stub + "*"):
            os.remove(file)
        for file in glob.glob(os.path.join("testing", "target_init_test_case_*")):
            os.remove(file)