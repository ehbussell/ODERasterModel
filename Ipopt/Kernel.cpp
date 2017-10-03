#include <cmath>

double kernel(int cell1, int cell2, int nrow, int ncol){
    double dist = 0.0;
    dist = sqrt(pow((cell1 % ncol) - (cell2 % ncol), 2) + pow((int)(cell1 / ncol) - (int)(cell2 / ncol), 2));

    return exp(-dist/0.1);

}