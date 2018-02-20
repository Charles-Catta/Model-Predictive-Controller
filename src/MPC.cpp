#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;


// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t eps_start = cte_start + N;
size_t delta_start = eps_start + N;
size_t a_start = delta_start + N - 1;

double max_vel = 50;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // ======= COST =========
    fg[0] = 0;

    for (size_t i = 0; i < N-1; i++) {
      fg[0] += 10 * pow(vars[a_start + i], 2);
      fg[0] += 10 * pow(vars[delta_start + i], 2);
    }

    for (size_t i = 0; i < N; i++) {
      fg[0] += pow(vars[v_start + i] - max_vel, 2);
      fg[0] += 2000 * pow(vars[eps_start + i], 2);
      fg[0] += 2000 * pow(vars[cte_start + i], 2);
    }

    for (size_t i = 0; i < N - 2; i++) {
      fg[0] += 10 * pow(vars[a_start + i + 1] - vars[a_start + i], 2);
      fg[0] += 100 * pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
    }

    // ======= INITIAL CONSTRAINTS =========
    fg[x_start + 1]    = vars[x_start];
    fg[y_start + 1]    = vars[y_start];
    fg[psi_start + 1]  = vars[psi_start];
    fg[v_start + 1]    = vars[v_start];
    fg[cte_start + 1]  = vars[cte_start];
    fg[eps_start + 1] = vars[eps_start];

    // ======= CONSTRAINTS =========
    for (size_t i = 1; i < N; i++) {
      // At time t
      AD<double> x_t0     = vars[x_start + i - 1];
      AD<double> y_t0     = vars[y_start + i - 1];
      AD<double> psi_t0   = vars[psi_start + i - 1];
      AD<double> v_t0     = vars[v_start + i - 1];
      AD<double> cte_t0   = vars[cte_start + i - 1];
      AD<double> eps_t0  = vars[eps_start + i - 1];
      AD<double> delta_t0 = vars[delta_start + i - 1];
      AD<double> a_t0     = vars[a_start + i - 1];

      AD<double> f_t0       = coeffs[0] + coeffs[1] * x_t0 + coeffs[2] * pow(x_t0, 2) + coeffs[3] * pow(x_t0, 3);
      AD<double> psi_des_t0 = atan(coeffs[1] + 2 * coeffs[2] * x_t0 + 3 * coeffs[3] * pow(x_t0, 2));
      
      // At time t + 1
      AD<double> x_t1    = vars[x_start + i];
      AD<double> y_t1    = vars[y_start + i];
      AD<double> psi_t1  = vars[psi_start + i];
      AD<double> v_t1    = vars[v_start + i];
      AD<double> cte_t1  = vars[cte_start + i];
      AD<double> eps_t1 = vars[eps_start + i];

      // Model constraints
      fg[x_start + i +1 ]    = x_t1    - (x_t0 + v_t0 * cos(psi_t0) * dt);
      fg[y_start + i + 1]    = y_t1    - (y_t0 + v_t0 * sin(psi_t0) * dt);
      fg[psi_start + i +1]   = psi_t1  - (psi_t0 - v_t0 * delta_t0 / Lf * dt);
      fg[v_start + i +1]     = v_t1    - (v_t0 + a_t0 * dt);
      fg[cte_start + i +1]   = cte_t1  - ((f_t0 - y_t0) + (v_t0 * sin(eps_t0) * dt));
      fg[eps_start + i + 1] = eps_t1 - ((psi_t0 - psi_des_t0) - v_t0 * delta_t0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  size_t n_vars = 6 * N + (N - 1) * 2;
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  double x    = state[0],
         y    = state[1],
         psi  = state[2],
         v    = state[3],
         cte  = state[4],
         eps  = state[5];

  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_upperbound(n_vars);
  Dvector vars_lowerbound(n_vars);

  for (size_t i = delta_start; i < a_start; i++) {
    vars_upperbound = deg2rad(25.0);
    vars_lowerbound = -deg2rad(25.0);
  }

  for (size_t i = a_start; i < n_vars; i++) {
    vars_upperbound[i] = 1.0;
    vars_lowerbound[i] = -1.0;
  }

  for (size_t i = 0; i < delta_start; i++) {
    vars_upperbound = 1.0e10;
    vars_lowerbound = -1.0e10;
  }
  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);

  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[eps_start] = eps;

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[eps_start] = eps;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  vector<double> solve_vec;

  solve_vec.push_back(solution.x[delta_start]);
  solve_vec.push_back(solution.x[a_start]);

  for (size_t i = 0; i < N; i++) {
    solve_vec.push_back(solution.x[x_start + i]);
    solve_vec.push_back(solution.x[y_start + i]);
  }
  return solve_vec;
}
