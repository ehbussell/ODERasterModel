#include "RasterModelEuler_NLP.hpp"
#include "Kernel.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

using namespace Ipopt;

// constructor
RasterModelEuler_NLP::RasterModelEuler_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments, std::vector<double> &init_state)
    : m_beta(beta),
    m_control_rate(control_rate),
    m_budget(budget),
    m_final_time(final_time),
    m_nrow(nrow),
    m_ncol(ncol),
    m_init_state(init_state),
    m_n_segments(n_segments),
    m_time_step(final_time / n_segments),
    m_ncells(nrow * ncol),
    m_warm_start(false)
{}

// constructor with warm start
RasterModelEuler_NLP::RasterModelEuler_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments, std::vector<double> &init_state, std::string start_file_stub)
    : RasterModelEuler_NLP(beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state)
{
    m_warm_start = true;
    m_start_file_stub = start_file_stub;
}

//destructor
RasterModelEuler_NLP::~RasterModelEuler_NLP()
{}

// Evaluate correct state indices
int RasterModelEuler_NLP::get_s_index(int time_idx, int space_idx)
{
    return 3*space_idx + time_idx*(3 * m_ncells);
}

int RasterModelEuler_NLP::get_i_index(int time_idx, int space_idx)
{
    return 1 + 3*space_idx + time_idx*(3 * m_ncells);
}

int RasterModelEuler_NLP::get_f_index(int time_idx, int space_idx)
{
    return 2 + 3*space_idx + time_idx*(3 * m_ncells);
}

// returns the size of the problem
bool RasterModelEuler_NLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    // Number of optimised variables
    n = 3 * m_ncells * (m_n_segments + 1);

    // Number of constraints
    m = (2*m_ncells + 1) * (m_n_segments + 1);

    // Number of non-zeros in Jacobian
    nnz_jac_g = m_ncells * (m_n_segments * (2*m_ncells + 7) + 4);

    // Number of non-zeros in Hessian - symmetric so only require lower left corner
    nnz_h_lag = m_ncells * (1 + 2*m_n_segments*(m_ncells + 1));

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    return true;
}

// returns the variable bounds
bool RasterModelEuler_NLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                Index m, Number* g_l, Number* g_u)
    {
    // all variables have lower bounds of 0
    for (Index i=0; i<n; i++) {
        x_l[i] = 0.0;
    }

    // all state variables have no upper bounds, control upper bound 1
    for (Index k=0; k<(m_n_segments+1); k++) {
        for (Index i=0; i<m_ncells; i++) {
            x_u[get_s_index(k, i)] = 2e19;
            x_u[get_i_index(k, i)] = 2e19;
            x_u[get_f_index(k, i)] = 1.0;
        }
    }

    // Continuity constraints
    for (Index i=0; i<(2*m_ncells*m_n_segments); i++){
        g_l[i] = g_u[i] = 0;
    }

    // Budget constraints
    for (Index i=0; i<(m_n_segments+1); i++){
        g_l[2 * m_ncells * m_n_segments + i] = 0.0;
        g_u[2 * m_ncells * m_n_segments + i] = m_budget;
    }

    // Initial conditions
    for (Index i=0; i<m_ncells; i++){
        g_l[2*m_ncells*m_n_segments + m_n_segments + 1 + i] = g_u[2*m_ncells*m_n_segments + m_n_segments + 1 + i] = m_init_state[get_s_index(0, i)];
        g_l[m_ncells*(2*m_n_segments + 1) + m_n_segments + 1 + i] = g_u[m_ncells*(2*m_n_segments + 1) + m_n_segments + 1 + i] = m_init_state[get_i_index(0, i)];
    }

    return true;
}

// returns the initial point for the problem
bool RasterModelEuler_NLP::get_starting_point(Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Index m, bool init_lambda,
                                   Number* lambda)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // initialize to the given starting point
    for (Index i=0; i<(3*m_ncells); i++){
        x[i] = m_init_state[i];
    }

    Index acc_idx = 3*m_ncells;
    double coupling_term;

    if (m_warm_start == false){
        // No warm start - initialise using euler method with no control
        for (Index k=0; k<m_n_segments; k++){
            for (Index i=0; i<m_ncells; i++){
                // Coupling term
                coupling_term = 0.0;
                for (Index j=0; j<m_ncells; j++){
                    coupling_term += kernel(i, j, m_nrow, m_ncol) * x[get_i_index(k, j)];
                }
                // Next S
                x[get_s_index(k+1, i)] = x[get_s_index(k, i)] * (1.0 - m_beta * coupling_term * m_time_step);
                acc_idx++;
                // Next I
                x[get_i_index(k+1, i)] = x[get_i_index(k, i)] + x[get_s_index(k, i)] * m_beta * coupling_term * m_time_step;
                acc_idx++;
                // Next f
                x[get_f_index(k+1, i)] = 0.0;
                acc_idx++;
            }
        }

        assert(acc_idx == n);
    } else {
        // Initialise from previous results files
        std::string tmp;
        // Read S input file
        std::ifstream inputFile(m_start_file_stub + "_S.csv");
        if (inputFile){
            std::getline(inputFile, tmp);

            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(inputFile, tmp, ',');
                    x[get_s_index(k, i)] = std::stod(tmp);
                }
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_S.csv" << std::endl;
            return false;
        }

        // Read I input file
        inputFile.open(m_start_file_stub + "_I.csv");
        if (inputFile){
            std::getline(inputFile, tmp);

            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(inputFile, tmp, ',');
                    x[get_i_index(k, i)] = std::stod(tmp);
                }
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_I.csv" << std::endl;
            return false;
        }

        // Read f input file
        inputFile.open(m_start_file_stub + "_f.csv");
        if (inputFile){
            std::getline(inputFile, tmp);

            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(inputFile, tmp, ',');
                    x[get_f_index(k, i)] = std::stod(tmp);
                }
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_f.csv" << std::endl;
            return false;
        }
    
    }


    return true;
}

// returns the value of the objective function
bool RasterModelEuler_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
    obj_value = 0.0;

    for (Index i=0; i<m_ncells; i++){
        obj_value -= x[get_s_index(m_n_segments, i)];
    }

    return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool RasterModelEuler_NLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
    for (Index i=0; i<n; i++){
        grad_f[i] = 0;
    }

    for (Index i=0; i<m_ncells; i++){
        grad_f[get_s_index(m_n_segments, i)] = -1.0;
    }

    return true;
}

// return the value of the constraints: g(x)
bool RasterModelEuler_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
    Index acc_idx = 0;
    double coupling_term;

    for (Index i=0; i<m_ncells; i++){

        // S and I continuity constraints
        for (Index k=0; k<m_n_segments; k++){
            // Calculate coupling term
            coupling_term = 0.0;
            for (Index j=0; j<m_ncells; j++){
                coupling_term += kernel(i, j, m_nrow, m_ncol) * x[get_i_index(k, j)];
            }

            // S constraint
            g[acc_idx] = x[get_s_index(k+1, i)] - x[get_s_index(k, i)] + (
                m_beta * x[get_s_index(k, i)] * coupling_term * m_time_step);
            acc_idx += 1;

                // I constraint
            g[acc_idx] = x[get_i_index(k+1, i)] - x[get_i_index(k, i)] - (
                m_beta * x[get_s_index(k, i)] * coupling_term -
                m_control_rate * x[get_f_index(k, i)] * x[get_i_index(k, i)]) * m_time_step;
            acc_idx += 1;
        }
    }

    assert(acc_idx == 2*m_ncells*m_n_segments);

    // Budget constraints
    for (Index k=0; k<(m_n_segments+1); k++){
        g[acc_idx] = 0.0;
        for (Index i=0; i<m_ncells; i++){
            g[acc_idx] += x[get_f_index(k, i)] * x[get_i_index(k, i)];
        }
        acc_idx += 1;
    }

    assert(acc_idx == (2*m_ncells*m_n_segments + m_n_segments + 1));

    // Initial Conditions
    for (Index i=0; i<m_ncells; i++){
        g[acc_idx] = x[get_s_index(0, i)];
        acc_idx += 1;
    }

    for (Index i=0; i<m_ncells; i++){
        g[acc_idx] = x[get_i_index(0, i)];
        acc_idx += 1;
    }

    assert(acc_idx == m);

    return true;
}

// return the structure or values of the jacobian
bool RasterModelEuler_NLP::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values)
{
    if (values == NULL) {
        // return the structure of the jacobian

        Index count = 0;
        Index constraint_count = 0;

        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k+1, i);
                count++;
                for (Index j=0; j<m_ncells; j++){
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k, j);
                    count++;
                }
                constraint_count++;
      
                // I continuity constraints
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k, i);
                count++;
                for (Index j=0; j<m_ncells; j++){
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k, j);
                    count++;
                }
                iRow[count] = constraint_count;
                jCol[count] = get_i_index(k+1, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_f_index(k, i);
                count++;
                constraint_count++;
            }
        }

        assert(constraint_count == 2*m_ncells*m_n_segments);
        assert(count == m_ncells*m_n_segments*(2*m_ncells + 5));

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            for (Index i=0; i<m_ncells; i++){
                iRow[count] = constraint_count;
                jCol[count] = get_i_index(k, i);
                count++;
            }
            for (Index i=0; i<m_ncells; i++){
                iRow[count] = constraint_count;
                jCol[count] = get_f_index(k, i);
                count++;
            }
            constraint_count++;
        }

        assert(constraint_count == 2*m_ncells*m_n_segments + m_n_segments + 1);
        assert(count == m_ncells*m_n_segments*(2*m_ncells + 5) + 2*m_ncells*(m_n_segments + 1));

        // Initial Conditions
        for (Index i=0; i<m_ncells; i++){
            iRow[count] = constraint_count;
            jCol[count] = get_s_index(0, i);
            count++;
            constraint_count++;
        }
    
        for (Index i=0; i<m_ncells; i++){
            iRow[count] = constraint_count;
            jCol[count] = get_i_index(0, i);
            count++;
            constraint_count++;
        }
    
        assert(count == nele_jac);

    }
    else {
        // return the values of the jacobian of the constraints

        Index count = 0;
        double coupling_term;
        
        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // Calculate coupling term
                coupling_term = 0.0;
                for (Index j=0; j<m_ncells; j++){
                    coupling_term += kernel(i, j, m_nrow, m_ncol) * x[get_i_index(k, j)];
                }

                // S continuity constraints
                values[count] = -1.0 + m_beta * coupling_term * m_time_step;
                count++;
                values[count] = 1.0;
                count++;
                for (Index j=0; j<m_ncells; j++){
                    values[count] = m_beta * x[get_s_index(k, i)] * kernel(i, j, m_nrow, m_ncol) * m_time_step;
                    count++;
                }
      
                // I continuity constraints
                values[count] = -m_beta * coupling_term * m_time_step;
                count++;
                for (Index j=0; j<m_ncells; j++){
                    values[count] = -m_beta * x[get_s_index(k, i)] * kernel(i, j, m_nrow, m_ncol) * m_time_step;
                    if (i == j){
                        values[count] += x[get_f_index(k, i)] * m_control_rate * m_time_step - 1.0;
                    }
                    count++;
                }
                values[count] = 1.0;
                count++;
                values[count] = m_control_rate * x[get_i_index(k, i)] * m_time_step;
                count++;
            }
        }

        assert(count == m_ncells*m_n_segments*(2*m_ncells + 5));

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            for (Index i=0; i<m_ncells; i++){
                values[count] = x[get_f_index(k, i)];
                count++;
            }
            for (Index i=0; i<m_ncells; i++){
                values[count] = x[get_i_index(k, i)];
                count++;
            }
        }

        // Initial Conditions
        for (Index i=0; i<m_ncells; i++){
            values[count] = 1.0;
            count++;
        }
    
        for (Index i=0; i<m_ncells; i++){
            values[count] = 1.0;
            count++;
        }
    
        assert(count == nele_jac);
    }

    return true;
}

//return the structure or values of the hessian
bool RasterModelEuler_NLP::eval_h(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values)
{
    if (values == NULL) {
        // return the structure. This is a symmetric matrix, fill the lower left triangle only.

        Index count = 0;
        
        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                for (Index j=0; j<m_ncells; j++){
                    if (i <= j){
                        iRow[count] = get_i_index(k, j);
                        jCol[count] = get_s_index(k, i);
                    } else {
                        iRow[count] = get_s_index(k, i);
                        jCol[count] = get_i_index(k, j);
                    }
                    assert(iRow[count] > jCol[count]);
                    count++;
                }
      
                // I continuity constraints
                for (Index j=0; j<m_ncells; j++){
                    if (i <= j){
                        iRow[count] = get_i_index(k, j);
                        jCol[count] = get_s_index(k, i);
                    } else {
                        iRow[count] = get_s_index(k, i);
                        jCol[count] = get_i_index(k, j);
                    }
                    assert(iRow[count] > jCol[count]);
                    count++;
                }
                iRow[count] = get_f_index(k, i);
                jCol[count] = get_i_index(k, i);
                assert(iRow[count] > jCol[count]);
                count++;
            }
        }

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            for (Index i=0; i<m_ncells; i++){
                iRow[count] = get_f_index(k, i);
                jCol[count] = get_i_index(k, i);
                count++;
            }
        }

        assert(count == nele_hess);

    }
    else {
        // return the values. This is a symmetric matrix, fill the lower left triangle only

        Index count = 0;
        Index constraint_count = 0;
        
        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                for (Index j=0; j<m_ncells; j++){
                    values[count] = lambda[constraint_count] * m_beta * kernel(i, j, m_nrow, m_ncol) * m_time_step;
                    count++;
                }
                constraint_count++;
      
                // I continuity constraints
                for (Index j=0; j<m_ncells; j++){
                    values[count] = -lambda[constraint_count] * m_beta * kernel(i, j, m_nrow, m_ncol) * m_time_step;
                    count++;
                }
                values[count] = lambda[constraint_count] * m_control_rate * m_time_step;
                count++;
                constraint_count++;
            }
        }

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            for (Index i=0; i<m_ncells; i++){
                values[count] = lambda[constraint_count];
                count++;
            }
            constraint_count++;
        }

        assert(count == nele_hess);
        assert(constraint_count == (m - 2*m_ncells));
    }

    return true;
}

void RasterModelEuler_NLP::finalize_solution(SolverReturn status,
                                  Index n, const Number* x, const Number* z_L, const Number* z_U,
                                  Index m, const Number* g, const Number* lambda,
                                  Number obj_value,
                                  const IpoptData* ip_data,
                                  IpoptCalculatedQuantities* ip_cq)
{
    // Write solution to files
    // Write S output file
    std::ofstream outputFile("output_S.csv");
    if (outputFile){
        outputFile << "time";
        for (Index i=0; i<m_ncells; i++){
            outputFile << ",Cell" << i;
        }
        outputFile << std::endl;
        for (Index i=0; i<(m_n_segments+1); i++){
            outputFile << i*m_time_step;
            for (Index j=0; j<m_ncells; j++){
                outputFile << "," << x[get_s_index(i, j)];
            }  
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Cannot open finalise file for writing - " << "output_S.csv" << std::endl;
    }

    // Write I output file
    outputFile.open("output_I.csv");
    if (outputFile){
        outputFile << "time";
        for (Index i=0; i<m_ncells; i++){
            outputFile << ",Cell" << i;
        }
        outputFile << std::endl;
        for (Index i=0; i<(m_n_segments+1); i++){
            outputFile << i*m_time_step;
            for (Index j=0; j<m_ncells; j++){
                outputFile << "," << x[get_i_index(i, j)];
            }  
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Cannot open finalise file for writing - " << "output_I.csv" << std::endl;
    }

    // Write f output file
    outputFile.open("output_f.csv");
    if (outputFile){
        outputFile << "time";
        for (Index i=0; i<m_ncells; i++){
            outputFile << ",Cell" << i;
        }
        outputFile << std::endl;
        for (Index i=0; i<(m_n_segments+1); i++){
            outputFile << i*m_time_step;
            for (Index j=0; j<m_ncells; j++){
                outputFile << "," << x[get_f_index(i, j)];
            }  
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Cannot open finalise file for writing - " << "output_f.csv" << std::endl;
    }

}
