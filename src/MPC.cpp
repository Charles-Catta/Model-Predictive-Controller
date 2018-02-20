#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;
double dt = 0.10;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
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
      fg[0] += 2000 * pow(vars[epsi_start + i], 2);
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
    fg[epsi_start + 1] = vars[epsi_start];

    // ======= CONSTRAINTS =========
    for (size_t i = 1; i < N; i++) {
      // At time t
      AD<double> x_t0     = vars[x_start + i - 1];
      AD<double> y_t0     = vars[y_start + i - 1];
      AD<double> psi_t0   = vars[psi_start + i - 1];
      AD<double> v_t0     = vars[v_start + i - 1];
      AD<double> cte_t0   = vars[cte_start + i - 1];
      AD<double> epsi_t0  = vars[epsi_start + i - 1];
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
      AD<double> epsi_t1 = vars[epsi_start + i];

      // Model constraints
      fg[x_start + i +1 ]    = x_t1    - (x_t0 + v_t0 * cos(psi_t0) * dt);
      fg[y_start + i + 1]    = y_t1    - (y_t0 + v_t0 * sin(psi_t0) * dt);
      fg[psi_start + i +1]   = psi_t1  - (psi_t0 - v_t0 * delta_t0 / Lf * dt);
      fg[v_start + i +1]     = v_t1    - (v_t0 + a_t0 * dt);
      fg[cte_start + i +1]   = cte_t1  - ((f_t0 - y_t0) + (v_t0 * sin(epsi_t0) * dt));
      fg[epsi_start + i + 1] = epsi_t1 - ((psi_t0 - psi_des_t0) - v_t0 * delta_t0 / Lf * dt);
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
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  size_t n_vars = 6 * N + (N - 1) * 2;
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

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

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x_t1 = {1.0,2.0}
  // creates a 2 element double vector.
  return {};
}
