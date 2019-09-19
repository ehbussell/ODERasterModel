#ifndef ConfigIncluded
#define ConfigIncluded
#include <string>

struct Config {
    int         method;
    double      beta;
    double      scale;
    double      control_rate;
    double      budget;
    double      final_time;
    int         n_segments;
    int         max_hosts;
    int         control_skip;
    int         non_spatial;
    int         control_start;
    std::string sus_file;
    std::string inf_file;
    std::string obj_file;
    std::string host_file_stub;
    std::string start_file_stub;

    Config(){
        method=1;
        budget=1.0;
        control_skip=0;
        non_spatial=0;
    }
};
#endif