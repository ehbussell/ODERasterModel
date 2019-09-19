#include "IpIpoptApplication.hpp"
#include "RasterModelEuler_nlp.hpp"
#include "RasterModelMidpoint_nlp.hpp"
#include "Kernel.hpp"
#include "RasterTools.hpp"
#include "Config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

using namespace Ipopt;

void loadConfig(Config& config, std::string config_file) {
    std::ifstream fin(config_file);
    std::string line;
    while (std::getline(fin, line)) {
        line.erase( std::find( line.begin(), line.end(), ';' ), line.end() );
        std::istringstream sin(line.substr(line.find("=") + 1));
        if (line.find("method") != -1)
            sin >> config.method;
        else if (line.find("beta") != -1)
            sin >> config.beta;
        else if (line.find("scale") != -1)
            sin >> config.scale;
        else if (line.find("control_rate") != -1)
            sin >> config.control_rate;
        else if (line.find("budget") != -1)
            sin >> config.budget;
        else if (line.find("final_time") != -1)
            sin >> config.final_time;
        else if (line.find("n_segments") != -1)
            sin >> config.n_segments;
        else if (line.find("max_hosts") != -1)
            sin >> config.max_hosts;
        else if (line.find("control_skip") != -1)
            sin >> config.control_skip;
        else if (line.find("non_spatial") != -1)
            sin >> config.non_spatial;
        else if (line.find("control_start") != -1)
            sin >> config.control_start;
        else if (line.find("sus_file") != -1)
            sin >> config.sus_file;
        else if (line.find("inf_file") != -1)
            sin >> config.inf_file;
        else if (line.find("obj_file") != -1)
            sin >> config.obj_file;
        else if (line.find("host_file_stub") != -1)
            sin >> config.host_file_stub;
        else if (line.find("start_file_stub") != -1)
            sin >> config.start_file_stub;
    }
}

int main(int argc, char* argv[])
{
    // Check correct number of arguments
    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " <CONFIG FILE>" << std::endl;
        return 1;
    }

    Config config;
    loadConfig(config, argv[1]);

    // Initialise model variables
    if (config.method != 0 && config.method != 1){
        std::cerr << "Method must be 0 for Euler, or 1 for Midpoint!" << config.method << std::endl;
        return 1;
    }

    // Extract initial state from rasters and save dimensions
    Raster host_raster = readRaster(config.host_file_stub + "HostDensity_raster.txt");
    Raster s0_raster = readRaster(config.host_file_stub + "S0_raster.txt");
    Raster i0_raster = readRaster(config.host_file_stub + "I0_raster.txt");
    Raster sus_raster = readRaster(config.sus_file);
    Raster inf_raster = readRaster(config.inf_file);

    int nrow, ncol;
    nrow = host_raster.m_nrows;
    ncol = host_raster.m_ncols;
    std::vector<double> init_state;
    std::vector<double> susceptibility;
    std::vector<double> infectiousness;

    for (int i=0; i<(nrow*ncol); i++){
        init_state.push_back(host_raster.m_array[i]*s0_raster.m_array[i]*config.max_hosts);
        init_state.push_back(host_raster.m_array[i]*i0_raster.m_array[i]*config.max_hosts);
        init_state.push_back(0.0);
        init_state.push_back(0.0);
        susceptibility.push_back(sus_raster.m_array[i]);
        infectiousness.push_back(sus_raster.m_array[i]);
    }

    // Read in objective weighting raster (terminal cell value)
    Raster obj_weights_raster = readRaster(config.obj_file);
    std::vector<double> obj_weights;
    for (int i=0; i<(nrow*ncol); i++){
        obj_weights.push_back(obj_weights_raster.m_array[i]);
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
    if (config.method == 0){
        // std::cout << std::endl << std::endl << "Using Euler method" << std::endl;
        // SmartPtr<RasterModelEuler_NLP> mynlp;
        // if (argc == 10){
        //     mynlp = new RasterModelEuler_NLP(
        //         beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state, argv[8]);
        // } else {
        //     mynlp = new RasterModelEuler_NLP(
        //         beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state);
        // }
        // status = app->OptimizeTNLP(mynlp);
    }
    if (config.method == 1){
        std::cout << std::endl << std::endl << "Using Midpoint method" << std::endl;
        SmartPtr<RasterModelMidpoint_NLP> mynlp;

        mynlp = new RasterModelMidpoint_NLP(
            config, nrow, ncol, init_state, obj_weights, susceptibility, infectiousness);

        status = app->OptimizeTNLP(mynlp);
    }
    
    if (status == Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }


    // Write setup output file
    std::ofstream outputFile("output.log");
    if (outputFile){
        outputFile << "METHOD " << config.method << std::endl;
        outputFile << "BETA " << config.beta << std::endl;
        outputFile << "SCALE " << config.scale << std::endl;
        outputFile << "CONTROL_RATE " << config.control_rate << std::endl;
        outputFile << "BUDGET " << config.budget << std::endl;
        outputFile << "FINAL_TIME " << config.final_time << std::endl;
        outputFile << "N_SEGMENTS " << config.n_segments << std::endl;
        outputFile << "MAX_HOSTS " << config.max_hosts << std::endl;
        outputFile << "CONTROL_SKIP " << config.control_skip << std::endl;
        outputFile << "NON_SPATIAL " << config.non_spatial << std::endl;
        outputFile << "CONTROL_START " << config.control_start << std::endl;
        outputFile << "SUS_FILE " << config.sus_file << std::endl;
        outputFile << "INF_FILE " << config.inf_file << std::endl;
        outputFile << "OBJ_FILE " << config.obj_file << std::endl;
        outputFile << "HOST_FILE_STUB " << config.host_file_stub << std::endl;
        outputFile << "START_FILE_STUB " << config.start_file_stub << std::endl;
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
