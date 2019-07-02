#include "RasterModelMidpoint_NLP.hpp"
#include "Kernel.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace Ipopt;

// constructor
RasterModelMidpoint_NLP::RasterModelMidpoint_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments,
                                                 std::vector<double> &init_state, std::vector<double> &obj_weights, int control_skip)
    : m_beta(beta),
    m_control_rate(control_rate),
    m_budget(budget),
    m_final_time(final_time),
    m_nrow(nrow),
    m_ncol(ncol),
    m_n_segments(n_segments),
    m_init_state(init_state),
    m_obj_weights(obj_weights),
    m_control_skip(control_skip),

    m_time_step(final_time / n_segments),
    m_ncells(nrow * ncol),
    m_warm_start(false),
    m_trunc_dist(2)
{}

// constructor with warm start
RasterModelMidpoint_NLP::RasterModelMidpoint_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments,
                                                 std::vector<double> &init_state, std::vector<double> &obj_weights, int control_skip, std::string start_file_stub)
: RasterModelMidpoint_NLP(beta, control_rate, budget, final_time, nrow, ncol, n_segments, init_state, obj_weights, control_skip)
{
m_warm_start = true;
m_start_file_stub = start_file_stub;
}

//destructor
RasterModelMidpoint_NLP::~RasterModelMidpoint_NLP()
{}

// Evaluate correct state indices
int RasterModelMidpoint_NLP::get_s_index(int time_idx, int space_idx)
{
    int prev_skip = (time_idx / (m_control_skip + 1)) * m_control_skip;
    if (time_idx % (m_control_skip + 1) == 0){
        // std::cout << "t: " << time_idx << " x: " << space_idx << " S: " << 4*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
        return 4*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells);
    } else {
        prev_skip += (time_idx - 1) % (m_control_skip + 1);
        // std::cout << "t: " << time_idx << " x: " << space_idx << " S: " << 2*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
        return 2*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells);
    }

}

int RasterModelMidpoint_NLP::get_i_index(int time_idx, int space_idx)
{
    int prev_skip = (time_idx / (m_control_skip + 1)) * m_control_skip;
    if (time_idx % (m_control_skip + 1) == 0){
        // std::cout << "I: " << 1 + 4*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
        return 1 + 4*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells);
    } else {
        prev_skip += (time_idx - 1) % (m_control_skip + 1);
        // std::cout << "I: " << 1 + 2*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
        return 1 + 2*space_idx + time_idx*(4 * m_ncells) - (prev_skip*2*m_ncells);
    }
}

int RasterModelMidpoint_NLP::get_u_index(int time_idx, int space_idx)
{
    int prev_skip = ((time_idx - (time_idx % (m_control_skip+1))) / (m_control_skip + 1)) * m_control_skip;
    // std::cout << "U: " << 2 + 4*space_idx + (time_idx - (time_idx % (m_control_skip+1)))*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
    return 2 + 4*space_idx + (time_idx - (time_idx % (m_control_skip+1)))*(4 * m_ncells) - (prev_skip*2*m_ncells);
}

int RasterModelMidpoint_NLP::get_v_index(int time_idx, int space_idx)
{
    int prev_skip = ((time_idx - (time_idx % (m_control_skip+1))) / (m_control_skip + 1)) * m_control_skip;
    // std::cout << "V: " << 3 + 4*space_idx + (time_idx - (time_idx % (m_control_skip+1)))*(4 * m_ncells) - (prev_skip*2*m_ncells) << std::endl;
    return 3 + 4*space_idx + (time_idx - (time_idx % (m_control_skip+1)))*(4 * m_ncells) - (prev_skip*2*m_ncells);
}

std::list<int> RasterModelMidpoint_NLP::get_connected(int space_idx)
{
    std::list<int> return_list;

    int start_row, end_row, start_col, end_col, idx;

    start_col = (space_idx % m_ncol) - m_trunc_dist;
    while (start_col < 0){
        start_col++;
    }
    end_col = (space_idx % m_ncol) + m_trunc_dist;
    while (end_col >= m_ncol){
        end_col--;
    }

    start_row = (int)(space_idx / m_ncol) - m_trunc_dist;
    while (start_row < 0){
        start_row++;
    }
    end_row = (int)(space_idx / m_ncol) + m_trunc_dist;
    while (end_row >= m_nrow){
        end_row--;
    }

    for (int row=start_row; row<(end_row+1); row++){
        for (int col=start_col; col<(end_col+1); col++){
            idx = (row * m_ncol) + col;
            return_list.push_back(idx);
        }
    }

    return return_list;
}

// returns the size of the problem
bool RasterModelMidpoint_NLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                             Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    // std::cout << "Starting get_nlp_info" << std::endl;

    // Number of optimised variables
    int prev_skip = (m_n_segments / (m_control_skip + 1)) * m_control_skip;
    if (m_n_segments % (m_control_skip + 1) != 0){
        prev_skip += m_n_segments % (m_control_skip + 1);
    }
    m_n_control_points = (m_n_segments + 1) - prev_skip;
    n = 4 * m_ncells * (m_n_segments + 1) - prev_skip * 2 * m_ncells;

    std::cout << "n: " << n << std::endl;

    // Number of constraints
    m = (2*m_ncells + 1) * (m_n_segments + 1) - prev_skip;

    // Number of non-zeros in Jacobian
    // TODO this
    // nnz_jac_g = 2 * m_ncells * (4 * m_n_segments + 2) - m_ncells * (m_n_segments + 1);
    nnz_jac_g = m_ncells * m_n_segments * 8 + m_n_control_points * m_ncells * 2 + m_ncells * 2;

    // Number of non-zeros in Hessian - symmetric so only require lower left corner
    // TODO this
    nnz_h_lag = 8 * m_ncells * m_n_segments;

    std::list<int> data;
    for (Index j=0; j<m_ncells; j++){
        data = get_connected(j);
        nnz_jac_g += 4 * m_n_segments * data.size();
        nnz_h_lag += 8 * m_n_segments * data.size();
    }

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    // std::cout << "Finished get_nlp_info" << std::endl;

    return true;
}

// returns the variable bounds
bool RasterModelMidpoint_NLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                Index m, Number* g_l, Number* g_u)
{
    // std::cout << "Starting get_bounds_info" << std::endl;
    // all variables have lower bounds of 0
    for (Index i=0; i<n; i++) {
        x_l[i] = 0.0;
    }

    // all state variables have no upper bounds, control upper bound 1
    for (Index k=0; k<(m_n_segments+1); k++) {
        for (Index i=0; i<m_ncells; i++) {
            x_u[get_s_index(k, i)] = 2e19;
            x_u[get_i_index(k, i)] = 2e19;
            x_u[get_u_index(k, i)] = 1.0;
            x_u[get_v_index(k, i)] = 1.0;
        }
    }

    // Continuity constraints
    for (Index i=0; i<(2*m_ncells*m_n_segments); i++){
        g_l[i] = g_u[i] = 0;
    }

    // Budget constraints
    for (Index i=0; i<(m_n_control_points); i++){
        g_l[2 * m_ncells * m_n_segments + i] = 0.0;
        g_u[2 * m_ncells * m_n_segments + i] = 1.0;
    }

    // Initial conditions
    for (Index i=0; i<m_ncells; i++){
        g_l[2*m_ncells*m_n_segments + m_n_control_points + i] = g_u[2*m_ncells*m_n_segments + m_n_control_points + i] = m_init_state[get_s_index(0, i)];
        g_l[m_ncells*(2*m_n_segments + 1) + m_n_control_points + i] = g_u[m_ncells*(2*m_n_segments + 1) + m_n_control_points + i] = m_init_state[get_i_index(0, i)];
    }

    // std::cout << "Finished get_bounds_info" << std::endl;

    return true;
}

// returns the initial point for the problem
bool RasterModelMidpoint_NLP::get_starting_point(Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Index m, bool init_lambda,
                                   Number* lambda)
{
    // std::cout << "Starting get_starting_point" << std::endl;
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // initialize to the given starting point
    for (Index i=0; i<(m_ncells); i++){
        x[get_s_index(0, i)] = m_init_state[4*i];
        x[get_i_index(0, i)] = m_init_state[4*i+1];
        x[get_u_index(0, i)] = m_init_state[4*i+2];
        x[get_v_index(0, i)] = m_init_state[4*i+3];
    }

    double coupling_term;
    std::list<int>::iterator it;
    std::list<int> data;

    if (m_warm_start == false){
        // No warm start - initialise using euler method with no control
        for (Index k=0; k<m_n_segments; k++){
            for (Index i=0; i<m_ncells; i++){
                // Coupling term
                coupling_term = 0.0;
                data = get_connected(i);
                for (it = data.begin(); it != data.end(); ++it){
                    coupling_term += kernel(i, *it, m_nrow, m_ncol) * x[get_i_index(k, *it)];
                }
                // Next S
                x[get_s_index(k+1, i)] = std::max(0.0,
                    x[get_s_index(k, i)] * (1.0 - m_beta * coupling_term * m_time_step)
                    - x[get_s_index(k, i)]*x[get_u_index(k, i)]*m_control_rate*m_time_step);
                // Next I
                x[get_i_index(k+1, i)] = std::max(0.0,
                    x[get_i_index(k, i)] + x[get_s_index(k, i)] * m_beta * coupling_term * m_time_step
                    - x[get_i_index(k, i)]*x[get_v_index(k, i)]*m_control_rate*m_time_step);
                // Next controls
                x[get_u_index(k+1, i)] = 0.0;
                x[get_v_index(k+1, i)] = 0.0;
            }
        }

    } else {
        // Initialise from previous results files
        std::string line;
        std::string tmp;
        std::stringstream iss;
        // Read S input file
        std::ifstream inputFile(m_start_file_stub + "_S.csv");
        if (inputFile){
            std::getline(inputFile, line);
            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, line);
                iss << line;
                std::getline(iss, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(iss, tmp, ',');
                    x[get_s_index(k, i)] = std::stod(tmp);
                }
                iss.clear();
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_S.csv" << std::endl;
            return false;
        }

        // Read I input file
        inputFile.open(m_start_file_stub + "_I.csv");
        if (inputFile){
            std::getline(inputFile, line);
            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, line);
                iss << line;
                std::getline(iss, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(iss, tmp, ',');
                    x[get_i_index(k, i)] = std::stod(tmp);
                }
                iss.clear();
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_I.csv" << std::endl;
            return false;
        }

        // Read u input file (thinning)
        inputFile.open(m_start_file_stub + "_u.csv");
        if (inputFile){
            std::getline(inputFile, line);
            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, line);
                iss << line;
                std::getline(iss, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(iss, tmp, ',');
                    x[get_u_index(k, i)] = std::stod(tmp);
                }
                iss.clear();
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_u.csv" << std::endl;
            return false;
        }

        // Read v input file (roguing)
        inputFile.open(m_start_file_stub + "_v.csv");
        if (inputFile){
            std::getline(inputFile, line);
            for (Index k=0; k<(m_n_segments+1); k++){
                std::getline(inputFile, line);
                iss << line;
                std::getline(iss, tmp, ',');
                for (Index i=0; i<m_ncells; i++){
                    std::getline(iss, tmp, ',');
                    x[get_v_index(k, i)] = std::stod(tmp);
                }
                iss.clear();
            }

            inputFile.close();
        } else {
            std::cerr << "Cannot open start file for reading - " << m_start_file_stub + "_v.csv" << std::endl;
            return false;
        }

    }

    // std::cout << "Finished get_starting_point" << std::endl;

    return true;
}

// returns the value of the objective function
bool RasterModelMidpoint_NLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
    // std::cout << "Starting eval_f" << std::endl;
    obj_value = 0.0;

    for (Index i=0; i<m_ncells; i++){
        obj_value -= x[get_s_index(m_n_segments, i)] * m_obj_weights[i];
    }

    // std::cout << "Finished eval_f" << std::endl;

    return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool RasterModelMidpoint_NLP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
    // std::cout << "Starting eval_grad_f" << std::endl;
    for (Index i=0; i<n; i++){
        grad_f[i] = 0;
    }

    for (Index i=0; i<m_ncells; i++){
        grad_f[get_s_index(m_n_segments, i)] = -m_obj_weights[i];
    }
    // std::cout << "Finished eval_grad_f" << std::endl;
    return true;
}

// return the value of the constraints: g(x)
bool RasterModelMidpoint_NLP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
    // std::cout << "Starting eval_g" << std::endl;
    Index acc_idx = 0;
    double coupling_term;
    std::list<int>::iterator it;
    std::list<int> data;

    for (Index i=0; i<m_ncells; i++){

        // S and I continuity constraints
        for (Index k=0; k<m_n_segments; k++){
            // Calculate coupling term
            coupling_term = 0.0;
            data = get_connected(i);
            for (it = data.begin(); it != data.end(); ++it){
                coupling_term += kernel(i, *it, m_nrow, m_ncol) * 0.5 * (x[get_i_index(k, *it)] + x[get_i_index(k+1, *it)]);
            }

            // S constraint
            g[acc_idx] = x[get_s_index(k+1, i)] - x[get_s_index(k, i)] + (
                m_beta * 0.5 * (x[get_s_index(k, i)] + x[get_s_index(k+1, i)]) *
                coupling_term + m_control_rate * 0.5 *
                (x[get_u_index(k, i)] + x[get_u_index(k+1, i)]) * 0.5 *
                (x[get_s_index(k, i)] + x[get_s_index(k+1, i)])) * m_time_step;
            acc_idx += 1;

            // I constraint
            g[acc_idx] = x[get_i_index(k+1, i)] - x[get_i_index(k, i)] - (
                m_beta * 0.5 * (x[get_s_index(k, i)] + x[get_s_index(k+1, i)]) * coupling_term
                - m_control_rate * 0.5 * (x[get_v_index(k, i)] + x[get_v_index(k+1, i)]) * 0.5 *
                (x[get_i_index(k, i)] + x[get_i_index(k+1, i)])) * m_time_step;
            acc_idx += 1;
        }
    }

    assert(acc_idx == 2*m_ncells*m_n_segments);

    // Budget constraints
    for (Index k=0; k<(m_n_segments+1); k++){
        if ((k % (m_control_skip+1)) == 0){
            g[acc_idx] = 0.0;
            for (Index i=0; i<m_ncells; i++){
                g[acc_idx] += x[get_u_index(k, i)] + x[get_v_index(k, i)];
            }
            acc_idx += 1;
        }
    }

    assert(acc_idx == (2*m_ncells*m_n_segments + m_n_control_points));

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

    // std::cout << "Finished eval_g" << std::endl;

    return true;
}

// return the structure or values of the jacobian
bool RasterModelMidpoint_NLP::eval_jac_g(Index n, const Number* x, bool new_x,
                           Index m, Index nele_jac, Index* iRow, Index *jCol,
                           Number* values)
{
    // std::cout << "Starting eval_jac_g" << std::endl;
    if (values == NULL) {
        // return the structure of the jacobian

        Index count = 0;
        Index constraint_count = 0;
        std::list<int>::iterator it;
        std::list<int> data;

        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k+1, i);
                count++;
                data = get_connected(i);
                for (it = data.begin(); it != data.end(); ++it){
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k, *it);
                    count++;
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k+1, *it);
                    count++;
                }
                iRow[count] = constraint_count;
                jCol[count] = get_u_index(k, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_u_index(k+1, i);
                count++;
                constraint_count++;

                // I continuity constraints
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_s_index(k+1, i);
                count++;
                for (it = data.begin(); it != data.end(); ++it){
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k, *it);
                    count++;
                    iRow[count] = constraint_count;
                    jCol[count] = get_i_index(k+1, *it);
                    count++;
                }
                iRow[count] = constraint_count;
                jCol[count] = get_v_index(k, i);
                count++;
                iRow[count] = constraint_count;
                jCol[count] = get_v_index(k+1, i);
                count++;
                constraint_count++;
            }
        }

        assert(constraint_count == 2*m_ncells*m_n_segments);

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            if ((k % (m_control_skip+1)) == 0){
                for (Index i=0; i<m_ncells; i++){
                    iRow[count] = constraint_count;
                    jCol[count] = get_u_index(k, i);
                    count++;
                    iRow[count] = constraint_count;
                    jCol[count] = get_v_index(k, i);
                    count++;
                }
                constraint_count++;
            }
        }

        // assert(constraint_count == 2*m_ncells*m_n_segments + m_n_segments + 1);

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
        std::list<int>::iterator it;
        std::list<int> data;

        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // Calculate coupling term
                coupling_term = 0.0;
                data = get_connected(i);
                for (it = data.begin(); it != data.end(); ++it){
                    coupling_term += kernel(i, *it, m_nrow, m_ncol) * 0.5 * (x[get_i_index(k, *it)] + x[get_i_index(k+1, *it)]);
                }

                // S continuity constraints
                values[count] = -1.0 + 0.5 * (m_beta * coupling_term * m_time_step);
                values[count] += 0.25 * (x[get_u_index(k, i)] + x[get_u_index(k+1, i)]) * m_control_rate * m_time_step;
                count++;
                values[count] = values[count-1] + 2;
                count++;
                for (it = data.begin(); it != data.end(); ++it){
                    values[count] = 0.5 * m_beta * 0.5 * (x[get_s_index(k, i)] + x[get_s_index(k+1, i)]) * kernel(i, *it, m_nrow, m_ncol) * m_time_step;
                    count++;
                    values[count] = values[count-1];
                    count++;
                }
                values[count] = 0.25 * m_control_rate * (x[get_s_index(k, i)] + x[get_s_index(k+1, i)]) * m_time_step;
                count++;
                values[count] = values[count-1];
                count++;

                // I continuity constraints
                values[count] = -0.5 * m_beta * coupling_term * m_time_step;
                count++;
                values[count] = values[count-1];
                count++;
                for (it = data.begin(); it != data.end(); ++it){
                    values[count] = -0.5 * m_beta * 0.5 * (x[get_s_index(k, i)] + x[get_s_index(k+1, i)]) * kernel(i, *it, m_nrow, m_ncol) * m_time_step;
                    if (i == *it){
                        values[count] -= 1.0;
                        values[count] += 0.25 * (x[get_v_index(k, i)] + x[get_v_index(k+1, i)]) * m_control_rate * m_time_step;
                    }
                    count++;
                    values[count] = values[count-1];
                    if (i == *it){
                        values[count] += 2.0;
                    }
                    count++;
                }
                values[count] = 0.25 * m_control_rate * (x[get_i_index(k, i)] + x[get_i_index(k+1, i)]) * m_time_step;
                count++;
                values[count] = values[count-1];
                count++;
            }
        }

        // Budget constraints
        for (Index k=0; k<(m_n_segments+1); k++){
            if ((k % (m_control_skip+1)) == 0){
                for (Index i=0; i<2*m_ncells; i++){
                    values[count] = 1.0;
                    count++;
                }
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

    // std::cout << "Finished eval_jac_g" << std::endl;

    return true;
}

//return the structure or values of the hessian
bool RasterModelMidpoint_NLP::eval_h(Index n, const Number* x, bool new_x,
                       Number obj_factor, Index m, const Number* lambda,
                       bool new_lambda, Index nele_hess, Index* iRow,
                       Index* jCol, Number* values)
{
    // std::cout << "Starting eval_h" << std::endl;
    if (values == NULL) {
        // return the structure. This is a symmetric matrix, fill the lower left triangle only.

        Index count = 0;
        std::list<int>::iterator it;
        std::list<int> data;

        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                data = get_connected(i);
                for (it = data.begin(); it != data.end(); ++it){
                    if (i <= *it){
                        iRow[count] = get_i_index(k, *it);
                        jCol[count] = get_s_index(k, i);
                        count++;
                        iRow[count] = get_i_index(k+1, *it);
                        jCol[count] = get_s_index(k+1, i);
                    } else {
                        iRow[count] = get_s_index(k, i);
                        jCol[count] = get_i_index(k, *it);
                        count++;
                        iRow[count] = get_s_index(k+1, i);
                        jCol[count] = get_i_index(k+1, *it);
                    }
                    count++;
                    iRow[count] = get_s_index(k+1, i);
                    jCol[count] = get_i_index(k, *it);
                    count++;
                    jCol[count] = get_i_index(k+1, *it);
                    iRow[count] = get_s_index(k, i);
                    count++;
                }
                if (get_u_index(k, i) <= get_s_index(k, i)){
                    iRow[count] = get_s_index(k, i);
                    jCol[count] = get_u_index(k, i);
                } else {
                    iRow[count] = get_u_index(k, i);
                    jCol[count] = get_s_index(k, i);
                }
                count++;
                if (get_u_index(k, i) <= get_s_index(k+1, i)){
                    iRow[count] = get_s_index(k+1, i);
                    jCol[count] = get_u_index(k, i);
                } else {
                    iRow[count] = get_u_index(k, i);
                    jCol[count] = get_s_index(k+1, i);
                }
                count++;
                if (get_u_index(k+1, i) <= get_s_index(k, i)){
                    iRow[count] = get_s_index(k, i);
                    jCol[count] = get_u_index(k+1, i);
                } else {
                    iRow[count] = get_u_index(k+1, i);
                    jCol[count] = get_s_index(k, i);
                }
                count++;
                if (get_u_index(k+1, i) <= get_s_index(k+1, i)){
                    iRow[count] = get_s_index(k+1, i);
                    jCol[count] = get_u_index(k+1, i);
                } else {
                    iRow[count] = get_u_index(k+1, i);
                    jCol[count] = get_s_index(k+1, i);
                }
                count++;

                // I continuity constraints
                for (it = data.begin(); it != data.end(); ++it){
                    if (i <= *it){
                        iRow[count] = get_i_index(k, *it);
                        jCol[count] = get_s_index(k, i);
                        count++;
                        iRow[count] = get_i_index(k+1, *it);
                        jCol[count] = get_s_index(k+1, i);
                    } else {
                        iRow[count] = get_s_index(k, i);
                        jCol[count] = get_i_index(k, *it);
                        count++;
                        iRow[count] = get_s_index(k+1, i);
                        jCol[count] = get_i_index(k+1, *it);
                    }
                    count++;
                    iRow[count] = get_s_index(k+1, i);
                    jCol[count] = get_i_index(k, *it);
                    count++;
                    iRow[count] = get_i_index(k+1, *it);
                    jCol[count] = get_s_index(k, i);
                    count++;
                }
                if (get_v_index(k, i) <= get_i_index(k, i)){
                    iRow[count] = get_i_index(k, i);
                    jCol[count] = get_v_index(k, i);
                } else {
                    iRow[count] = get_v_index(k, i);
                    jCol[count] = get_i_index(k, i);
                }
                count++;
                if (get_v_index(k, i) <= get_i_index(k+1, i)){
                    iRow[count] = get_i_index(k+1, i);
                    jCol[count] = get_v_index(k, i);
                } else {
                    iRow[count] = get_v_index(k, i);
                    jCol[count] = get_i_index(k+1, i);
                }
                count++;
                if (get_v_index(k+1, i) <= get_i_index(k, i)){
                    iRow[count] = get_i_index(k, i);
                    jCol[count] = get_v_index(k+1, i);
                } else {
                    iRow[count] = get_v_index(k+1, i);
                    jCol[count] = get_i_index(k, i);
                }
                count++;
                if (get_v_index(k+1, i) <= get_i_index(k+1, i)){
                    iRow[count] = get_i_index(k+1, i);
                    jCol[count] = get_v_index(k+1, i);
                } else {
                    iRow[count] = get_v_index(k+1, i);
                    jCol[count] = get_i_index(k+1, i);
                }
                count++;
            }
        }

        assert(count == nele_hess);

    }
    else {
        // return the values. This is a symmetric matrix, fill the lower left triangle only

        Index count = 0;
        Index constraint_count = 0;
        std::list<int>::iterator it;
        std::list<int> data;

        for (Index i=0; i<m_ncells; i++){
            for (Index k=0; k<m_n_segments; k++){
                // S continuity constraints
                data = get_connected(i);
                for (it = data.begin(); it != data.end(); ++it){
                    values[count] = lambda[constraint_count] * 0.25 * m_beta * kernel(i, *it, m_nrow, m_ncol) * m_time_step;;
                    count++;
                    values[count] = values[count-1];
                    count++;
                    values[count] = values[count-1];
                    count++;
                    values[count] = values[count-1];
                    count++;
                }
                values[count] = lambda[constraint_count] * 0.25 * m_control_rate * m_time_step;
                count++;
                values[count] = values[count-1];
                count++;
                values[count] = values[count-1];
                count++;
                values[count] = values[count-1];
                count++;
                constraint_count++;

                // I continuity constraints
                for (it = data.begin(); it != data.end(); ++it){
                    values[count] = -lambda[constraint_count] * 0.25 * m_beta * kernel(i, *it, m_nrow, m_ncol) * m_time_step;;
                    count++;
                    values[count] = values[count-1];
                    count++;
                    values[count] = values[count-1];
                    count++;
                    values[count] = values[count-1];
                    count++;
                }
                values[count] = lambda[constraint_count] * 0.25 * m_control_rate * m_time_step;
                count++;
                values[count] = values[count-1];
                count++;
                values[count] = values[count-1];
                count++;
                values[count] = values[count-1];
                count++;
                constraint_count++;
            }
        }

        assert(count == nele_hess);
        assert(constraint_count == (m - 2*m_ncells - m_n_control_points));
    }

    // std::cout << "Finished eval_h" << std::endl;

    return true;
}

void RasterModelMidpoint_NLP::finalize_solution(SolverReturn status,
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

    // Write u output file
    outputFile.open("output_u.csv");
    if (outputFile){
        outputFile << "time";
        for (Index i=0; i<m_ncells; i++){
            outputFile << ",Cell" << i;
        }
        outputFile << std::endl;
        for (Index i=0; i<(m_n_segments+1); i++){
            outputFile << i*m_time_step;
            for (Index j=0; j<m_ncells; j++){
                outputFile << "," << x[get_u_index(i, j)];
            }
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Cannot open finalise file for writing - " << "output_u.csv" << std::endl;
    }

    // Write v output file
    outputFile.open("output_v.csv");
    if (outputFile){
        outputFile << "time";
        for (Index i=0; i<m_ncells; i++){
            outputFile << ",Cell" << i;
        }
        outputFile << std::endl;
        for (Index i=0; i<(m_n_segments+1); i++){
            outputFile << i*m_time_step;
            for (Index j=0; j<m_ncells; j++){
                outputFile << "," << x[get_v_index(i, j)];
            }
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Cannot open finalise file for writing - " << "output_v.csv" << std::endl;
    }

}
