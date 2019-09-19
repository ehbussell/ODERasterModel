#include "IpTNLP.hpp"
#include "Config.hpp"

using namespace Ipopt;

/* C++ test for optimal control of RasterModel model - Midpoint method for discretisation 
 *  
 */

class RasterModelMidpoint_NLP : public TNLP
{
public:
    /** default constructor */
    RasterModelMidpoint_NLP(Config config, int nrow, int ncol, std::vector<double> &init_state,
                            std::vector<double> &obj_weights, std::vector<double> &susceptibility,
                            std::vector<double> &infectiousness);
    // RasterModelMidpoint_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments,
    //                         std::vector<double> &init_state, std::vector<double> &obj_weights, int control_skip,
    //                         std::vector<double> &susceptibility, std::vector<double> &infectiousness);
    
    // /** constructor with warm start point */
    // RasterModelMidpoint_NLP(double beta, double control_rate, double budget, double final_time, int nrow, int ncol, int n_segments,
    //                         std::vector<double> &init_state, std::vector<double> &obj_weights, int control_skip,
    //                         std::vector<double> &susceptibility, std::vector<double> &infectiousness, std::string start_file_stub);

    /** default destructor */
    virtual ~RasterModelMidpoint_NLP();

    double m_beta, m_control_rate, m_budget, m_final_time, m_time_step, m_scale;
    int m_nrow, m_ncol, m_ncells, m_n_segments, m_trunc_dist, m_control_skip, m_n_control_points, m_non_spatial, m_control_start;
    std::vector<double> m_init_state, m_infectiousness, m_susceptibility;
    bool m_warm_start;
    std::string m_start_file_stub;
    std::vector<double> m_obj_weights;

    int get_s_index(int time_idx, int space_idx);
    int get_i_index(int time_idx, int space_idx);
    int get_u_index(int time_idx, int space_idx);
    int get_v_index(int time_idx, int space_idx);
     std::list<int> get_connected(int space_idx);

    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style);

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                 Index m, Number* g_l, Number* g_u);

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda,
                                    Number* lambda);

    /** Method to return the objective value */
    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);

    /** Method to return the constraint residuals */
    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

    /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow, Index *jCol,
                            Number* values);

    /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
    virtual bool eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess, Index* iRow,
                        Index* jCol, Number* values);

    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value, const IpoptData* ip_data, 
                                   IpoptCalculatedQuantities* ip_cq);

};