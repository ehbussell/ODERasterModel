#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include "RasterTools.hpp"

/** default constructor */
Raster::Raster(int ncols, int nrows, double xllcorner, double yllcorner, int cellsize, double NODATA_value, std::vector<double> array)
    : m_ncols(ncols),
    m_nrows(nrows),
    m_xllcorner(xllcorner),
    m_yllcorner(yllcorner),
    m_cellsize(cellsize),
    m_NODATA_value(NODATA_value),
    m_array(array)
{
    if (m_ncols*m_nrows != array.size()){
        std::cerr << "Array not correct size!" << std::endl;
    }
}

/** default destructor */
Raster::~Raster(){}

Raster readRaster(std::string inFileName){

    int ncols, nrows, cellsize;
    double xllcorner, yllcorner, NODATA_value;
    std::vector<double> array;

    std::ifstream inputFile(inFileName);
    if (inputFile){
        std::string tmp_s;
        double tmp_d;

        // Read header
        for (int i=0; i<6; i++){
            inputFile >> tmp_s;
            std::transform(tmp_s.begin(), tmp_s.end(), tmp_s.begin(),(int (*)(int))std::toupper);
            if (tmp_s.compare("NCOLS") == 0){
                inputFile >> ncols;
            }
            if (tmp_s.compare("NROWS") == 0){
                inputFile >> nrows;
            }
            if (tmp_s.compare("XLLCORNER") == 0){
                inputFile >> xllcorner;
            }
            if (tmp_s.compare("YLLCORNER") == 0){
                inputFile >> yllcorner;
            }
            if (tmp_s.compare("CELLSIZE") == 0){
                inputFile >> cellsize;
            }
            if (tmp_s.compare("NODATA_VALUE") == 0){
                inputFile >> NODATA_value;
            }
        }

        // Read array
        for (int i=0; i<(nrows*ncols); i++){
            inputFile >> tmp_d;
            array.push_back(tmp_d);
        }

        // Close
        inputFile.close();
    } else {
        std::cerr << "Cannot open file - " << inFileName << std::endl;
    }

    Raster retRaster = Raster(ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value, array);

    return retRaster;
}

void writeRaster(std::string outFileName, Raster raster){
    // Do stuff...
}