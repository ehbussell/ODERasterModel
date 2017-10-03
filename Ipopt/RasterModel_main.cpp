#include "IpIpoptApplication.hpp"
#include "RasterModel_nlp.hpp"
#include "Kernel.hpp"
#include "RasterTools.hpp"
#include <fstream>

#include <iostream>

using namespace Ipopt;

int main(int argc, char* argv[])
{
    // Check correct number of arguments
    if (argc != 7){
        std::cerr << "Usage: " << argv[0] << " <BETA> <CONTROL RATE> <BUDGET> <FINAL TIME> <N SEGMENTS> <MAX HOSTS>" << std::endl;
        return 1;
    }

    // Initialise model variables
    double beta, control_rate, budget, final_time;
    beta = std::stof(argv[1]);
    control_rate = std::stof(argv[2]);
    budget = std::stof(argv[3]);
    final_time = std::stof(argv[4]);

    int n_segments, max_hosts;
    n_segments = std::stoi(argv[5]);
    max_hosts = std::stoi(argv[6]);

    // Extract initial state from rasters and save dimensions TODO!!!
    Raster host_raster = readRaster("HostDensity_raster.txt");
    Raster s0_raster = readRaster("S0_raster.txt");
    Raster i0_raster = readRaster("I0_raster.txt");

    int nrow, ncol;
    nrow = host_raster.m_nrows;
    ncol = host_raster.m_ncols;
    std::vector<double> init_state;

    for (int i=0; i<(nrow*ncol); i++){
        init_state.push_back(host_raster.m_array[i]*s0_raster.m_array[i]*max_hosts);
        init_state.push_back(host_raster.m_array[i]*i0_raster.m_array[i]*max_hosts);
        init_state.push_back(0.0);
    }

    // Create a new instance of nlp
    //  (use a SmartPtr, not raw)
    SmartPtr<RasterModel_NLP> mynlp = new RasterModel_NLP(
        beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state);

    // Create a new instance of IpoptApplication
    //  (use a SmartPtr, not raw)
    // We are using the factory, since this allows us to compile this
    // example with an Ipopt Windows DLL
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->RethrowNonIpoptException(true);

    // Change some options
    // Note: The following choices are only examples, they might not be
    //       suitable for your optimization problem.
    app->Options()->SetNumericValue("tol", 1e-7);
    app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", "ipopt.out");
    // The following overwrites the default name (ipopt.opt) of the
    // options file
    // app->Options()->SetStringValue("option_file_name", "h.opt");

    // Initialize the IpoptApplication and process the options
    ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return (int) status;
    }

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(mynlp);

    if (status == Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }

    // As the SmartPtrs go out of scope, the reference count
    // will be decremented and the objects will automatically
    // be deleted.

    return (int) status;
}
