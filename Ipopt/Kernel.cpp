#include <cmath>

double dist = 0.0;
const double pi = 3.1415926535897;

double kernel(int cell1, int cell2, int nrow, int ncol, double sus, double inf, double scale){
    dist = sqrt(pow((cell1 % ncol) - (cell2 % ncol), 2) + pow((int)(cell1 / ncol) - (int)(cell2 / ncol), 2));

    // Exponential kernel
    // return sus * inf * exp(-dist/scale) / (2 * pi * scale * scale);

    // Cauchy kernel
    return sus * inf * 2 / (pi * scale * (1 + pow(dist/scale, 2)));

}

double kernel(int cell1, int cell2, int nrow, int ncol, double scale){
    dist = sqrt(pow((cell1 % ncol) - (cell2 % ncol), 2) + pow((int)(cell1 / ncol) - (int)(cell2 / ncol), 2));

    return exp(-dist/scale) / (2 * pi * scale * scale);

}