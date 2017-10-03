
class Raster
{
public:
    /** default constructor */
    Raster(int ncols, int nrows, double xllcorner, double yllcorner, int cellsize, double NODATA_value, std::vector<double> array);

    /** default destructor */
    virtual ~Raster();

    int m_ncols, m_nrows, m_cellsize;
    double m_xllcorner, m_yllcorner, m_NODATA_value;
    std::vector<double> m_array;

    
};

Raster readRaster(std::string inFileName);

void writeRaster(std::string inFileName, Raster raster);