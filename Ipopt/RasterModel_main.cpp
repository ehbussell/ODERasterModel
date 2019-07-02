#include "IpIpoptApplication.hpp"
#include "RasterModelEuler_nlp.hpp"
#include "RasterModelMidpoint_nlp.hpp"
#include "Kernel.hpp"
#include "RasterTools.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace Ipopt;

int main(int argc, char* argv[])
{
    // Check correct number of arguments
    if (argc != 9 && argc != 10){
        std::cerr << "Usage: " << argv[0] << " <METHOD> <BETA> <CONTROL RATE> <BUDGET> <FINAL TIME> <N SEGMENTS> <MAX HOSTS> <CONTROL SKIP> <START FILE STUB>" << std::endl;
        return 1;
    }

    // Initialise model variables
    int discretise_method;
    discretise_method = std::stoi(argv[1]);
    if (discretise_method != 0 && discretise_method != 1){
        std::cerr << "Method must be 0 for Euler, or 1 for Midpoint!" << discretise_method << std::endl;
        return 1;
    }

    double beta, control_rate, budget, final_time;
    beta = std::stod(argv[2]);
    control_rate = std::stod(argv[3]);
    budget = std::stod(argv[4]);
    final_time = std::stod(argv[5]);

    int n_segments, max_hosts;
    n_segments = std::stoi(argv[6]);
    max_hosts = std::stoi(argv[7]);
    int control_skip = std::stoi(argv[8]);

    // Extract initial state from rasters and save dimensions
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

    // Create a new instance of IpoptApplication
    //  (use a SmartPtr, not raw)
    // We are using the factory, since this allows us to compile this
    // example with an Ipopt Windows DLL
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->RethrowNonIpoptException(true);

    // Change some options
    // Note: The following choices are only examples, they might not be
    //       suitable for your optimization problem.
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

    // Create a new instance of nlp
    //  (use a SmartPtr, not raw)
    // Ask Ipopt to solve the problem
    if (discretise_method == 0){
        std::cout << std::endl << std::endl << "Using Euler method" << std::endl;
        SmartPtr<RasterModelEuler_NLP> mynlp;
        if (argc == 10){
            mynlp = new RasterModelEuler_NLP(
                beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state, argv[8]);
        } else {
            mynlp = new RasterModelEuler_NLP(
                beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state);
        }
        status = app->OptimizeTNLP(mynlp);
    }
    if (discretise_method == 1){
        std::cout << std::endl << std::endl << "Using Midpoint method" << std::endl;
        SmartPtr<RasterModelMidpoint_NLP> mynlp;
        if (argc == 10){
            mynlp = new RasterModelMidpoint_NLP(
                beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state, control_skip, argv[9]);
        } else {
            mynlp = new RasterModelMidpoint_NLP(
                beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state, control_skip);
        }
        status = app->OptimizeTNLP(mynlp);
    }
    
    if (status == Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }

    std::vector<std::string> arg_names = {
        "METHOD",
        "BETA",
        "CONTROL_RATE",
        "BUDGET",
        "FINAL_TIME",
        "N_SEGMENTS",
        "MAX_HOSTS",
        "START_FILE_STUB"
    };

    // Write setup output file
    std::ofstream outputFile("output.log");
    if (outputFile){
        for (Index i=0; i<(argc-1); i++){
            outputFile << arg_names[i] << " " << argv[i+1] << std::endl;
        }
        outputFile << "EXIT_CODE " << status << std::endl;
        outputFile.close();
    } else {
        std::cerr << "Cannot open log file for writing - " << "output.log" << std::endl;
    }

    // As the SmartPtrs go out of scope, the reference count
    // will be decremented and the objects will automatically
    // be deleted.

    return (int) status;
}
