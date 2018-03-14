import os
import numpy as np
import pandas as pd
import unittest
import raster_model_fitting as rmf
import raster_tools

class QuickTests(unittest.TestCase):
    """Tests to check setup and running of raster_model_fitting module."""

    @classmethod
    def setUpClass(cls):
        # Setup fake simulation data
        cls._data_stub = os.path.join("testing", "quick_sim_output")

        # Create host file
        host_raster = raster_tools.RasterData((2, 2), array=np.ones((2, 2)))
        host_file = os.path.join("testing", "host_test_case.txt")
        host_raster.to_file(host_file)

        # Create initial conditions files
        init_stub = os.path.join("testing", "init_test_case")
        host_raster.array[1, 1] = 0
        host_raster.to_file(init_stub + "_S.txt")
        host_raster.array = np.zeros((2, 2))
        host_raster.array[1, 1] = 1
        host_raster.to_file(init_stub + "_I.txt")

        # Create log file
        log_str = "Configuration File Used\n" + "#"*20 + "\n"
        log_str += "\n[Epidemiology]\n"
        log_str += "Model = SI\nInfRate = 0.8\nKernelType = RASTER\n"
        log_str += "\n[Simulation]\n"
        log_str += "SimulationType = RASTER\nFinalTime = 10.0\nNIterations = 2\n"
        log_str += "HostPosFile = " + host_file + "\nInitCondFile = " + init_stub + "\n"
        log_str += "\n[Output]\n"
        log_str += "SummaryOutputFreq = 0\nOutputFileStub = " + cls._data_stub
        with open(cls._data_stub + ".log", "w") as outfile:
            outfile.write(log_str)

        # Create host data files
        host_data_dict = {
            "posX": np.tile(np.arange(0.5, 2.5), 2),
            "posY": np.repeat(np.arange(1.5, -0.5, -1), 2),
            "hostID": list(range(4)),
            "initial_state": ["S"]*3 + ["I"]
        }
        host_data = pd.DataFrame(host_data_dict)
        for i in range(2):
            filename = cls._data_stub + "_hosts_" + str(i) + ".csv"
            host_data.to_csv(filename, index=False)

        # Create event data files
        cls._inf_hosts = np.array([
            [0, 1, 2],
            [1, 2, 0]
        ])
        for i in range(2):
            event_data_dict = {
                "time": [2, 4, 6],
                "hostID": cls._inf_hosts[i]
            }
            event_data = pd.DataFrame(event_data_dict)
            filename = cls._data_stub + "_events_" + str(i) + ".csv"
            event_data.to_csv(filename, index=False)

        cls._raster_header = host_raster.header_vals

    def test_full_precalc(self):
        """Test full likelihood precalculation."""
        correct_const_factors = np.array([0, 0, -16, -12, -12])
        correct_matrix = np.array([
            [1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ])
        correct_distances = np.array([0, 1, 1, np.sqrt(2)])

        lik_func = rmf.precompute_loglik(data_stub=self._data_stub, nsims=None,
                                         raster_header=self._raster_header, precompute_level="full")

        print(lik_func.matrix)

        self.assertEqual(lik_func.precompute_level, "full")
        self.assertTrue(np.allclose(lik_func.const_factors, correct_const_factors))
        self.assertTrue(np.allclose(lik_func.matrix, correct_matrix))
        self.assertTrue(np.allclose(lik_func.distances, correct_distances))

    def test_partial_precalc(self):
        """Test partial likelihood precalculation."""
        correct_const_factors = np.array([0, 0, -16, -12, -12])
        correct_distances = np.array([0, 1, 1, np.sqrt(2)])
        correct_inf_cell_ids = np.array([
            [0, 1, 2],
            [1, 2, 0]
        ])
        correct_init_inf = np.array([0, 0, 0, 1])
        correct_rel_pos_array = np.array([
            [0, 1, 1, np.sqrt(2)],
            [1, 0, np.sqrt(2), 1],
            [1, np.sqrt(2), 0, 1],
            [np.sqrt(2), 1, 1, 0]
        ])

        lik_func = rmf.precompute_loglik(
            data_stub=self._data_stub, nsims=None, raster_header=self._raster_header,
            precompute_level="partial")

        print(lik_func.all_inf_cell_ids)

        self.assertTrue(lik_func.precompute_level == "partial")
        self.assertTrue(np.allclose(lik_func.const_factors, correct_const_factors))
        self.assertTrue(np.allclose(lik_func.distances, correct_distances))
        self.assertTrue(np.allclose(lik_func.all_inf_cell_ids, correct_inf_cell_ids))
        self.assertTrue(np.allclose(lik_func.initial_inf, correct_init_inf))
        self.assertTrue(np.allclose(lik_func.rel_pos_array, correct_rel_pos_array))

    def test_full_evaluation(self):
        """Test full likelihood evaluation."""
        const_factors = np.random.rand(50)
        matrix = np.random.rand(200, 50)
        distances = np.random.rand(49)
        primary_factor = np.random.random_sample()

        data_dict = {
            "const_factors": const_factors,
            "matrix": matrix,
            "distances": distances,
            "primary_factor": primary_factor
        }

        lik_func = rmf.LikelihoodFunction("full", data_dict)

        kernel = lambda d: np.exp(-d)

        epsilon = np.random.random_sample()

        k_vals = np.array([1] + list(kernel(distances)))
        correct_value = np.dot(const_factors, k_vals)
        k_vals[0] = epsilon
        correct_value += np.sum(np.log(np.dot(matrix, k_vals)))
        correct_value += primary_factor * epsilon

        self.assertAlmostEqual(lik_func.eval_loglik(kernel, primary_rate=epsilon)[0], correct_value)

    def test_partial_evaluation(self):
        """Test partial likelihood evaluation."""
        const_factors = np.random.rand(50)
        distances = np.random.rand(49)
        primary_factor = np.random.random_sample()
        inf_cell_ids = [np.random.choice(15, 10, replace=False)]
        init_inf = np.zeros(16)
        init_inf[15] = 1
        rel_pos_array = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                i_pos = np.unravel_index(i, (16, 1))
                j_pos = np.unravel_index(j, (16, 1))
                rel_pos = (abs(i_pos[0] - j_pos[0]), abs(i_pos[1] - j_pos[1]))
                rel_pos_array[i, j] = np.sqrt(rel_pos[0]*rel_pos[0] + rel_pos[1]*rel_pos[1])

        data_dict = {
            "distances": distances,
            "const_factors": const_factors,
            "all_inf_cell_ids": inf_cell_ids,
            "initial_inf": init_inf,
            "rel_pos_array": rel_pos_array,
            "primary_factor": primary_factor
        }

        lik_func = rmf.LikelihoodFunction("partial", data_dict)

        kernel = lambda d: np.exp(-d)
        epsilon = np.random.random_sample()
        lik_value = lik_func.eval_loglik(kernel, primary_rate=epsilon)[0]

        k_vals = np.array([1] + list(kernel(distances)))
        correct_value = np.dot(const_factors, k_vals)
        for cell_id in inf_cell_ids[0]:
            correct_value += np.log(epsilon + np.sum([
                init_inf[x]*kernel(rel_pos_array[x, cell_id]) for x in range(16)]))
            init_inf[cell_id] += 1
        correct_value += primary_factor * epsilon

        self.assertAlmostEqual(lik_value, correct_value)

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join("testing", "host_test_case.txt"))
        os.remove(os.path.join("testing", "init_test_case_S.txt"))
        os.remove(os.path.join("testing", "init_test_case_I.txt"))
        os.remove(os.path.join(cls._data_stub + ".log"))
        for i in range(2):
            os.remove(os.path.join(cls._data_stub + "_hosts_" + str(i) +".csv"))
            os.remove(os.path.join(cls._data_stub + "_events_" + str(i) +".csv"))