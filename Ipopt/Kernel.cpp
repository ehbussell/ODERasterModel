#include <cmath>

double kernel(int cell1, int cell2, int nrow, int ncol){
    double dist = 0.0;
    double scale = 0.294171;
    const double pi = 3.1415926535897;
    dist = sqrt(pow((cell1 % ncol) - (cell2 % ncol), 2) + pow((int)(cell1 / ncol) - (int)(cell2 / ncol), 2));

    return exp(-dist/scale) / (2 * pi * scale * scale);

}