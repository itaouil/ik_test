#include <iostream>
#include <Eigen/Geometry>
#include "../matplotlib-cpp/matplotlibcpp.h"

using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;
using trafo2d_t = Eigen::Transform<double, 2, Eigen::Affine>;

void plot_q(vector_t const &q_start, vector_t const &q_ik, std::string const &plot_title) {
    // IK EE joint segments
    std::vector<double> x_ik(4);
    std::vector<double> y_ik(4);

    // Start EE joint segments
    std::vector<double> x_start(4);
    std::vector<double> y_start(4);

    // Base
    x_ik.at(0) = 0;
    y_ik.at(0) = 0;
    x_start.at(0) = 0;
    y_start.at(0) = 0;

    // Rest of the segments
    double ik_angle = 0.0;
    double start_angle = 0.0;
    for(int i=1; i<4; i++) {
        // Aggregate angles
        ik_angle += (q_ik(i-1) * -1);
        start_angle += (q_start(i-1) * -1);

        // Populate start EE segments
        y_start.at(i) = cos(start_angle) + y_start.at(i-1);
        x_start.at(i) = sin(start_angle) + x_start.at(i-1);

        // Populate IK EE segments
        y_ik.at(i) = cos(ik_angle) + y_ik.at(i-1);
        x_ik.at(i) = sin(ik_angle) + x_ik.at(i-1);
    }

    // Plot
    matplotlibcpp::named_plot("Start Configuration", x_start, y_start);
    matplotlibcpp::named_plot("IK Configuration", x_ik, y_ik);
    matplotlibcpp::title(plot_title);
    matplotlibcpp::legend();
    matplotlibcpp::show();
}

/**
 * Function computing the Jacobian
 * matrix of a a planar 3-DOF manipulator.
 *
 * @param q
 * @return 2x3 Jacobian matrix
 */
matrix_t jacobian(vector_t const &q) {
    // Jacobian matrix to be populated
    matrix_t j(2, 3);

    // Prepare angles
    auto a0 = M_PI_2 + (q(0) * -1);
    auto a1 = M_PI_2 + (q(1) * -1);
    auto a2 = M_PI_2 + (q(2) * -1);

    // First joint column vector of partial derivatives
    j.block<2, 1>(0 ,0) = vector_t{{-sin(a0) - sin(a0+a1) - sin(a0+a1+a2)},
                                                    {cos(a0) + cos(a0+a1) + cos(a0+a1+a2)}};

    // Second joint column vector of partial derivatives
    j.block<2, 1>(0 ,1) = vector_t{{-sin(a0+a1) - sin(a0+a1+a2)},
                                                    {cos(a0+a1) + cos(a0+a1+a2)}};

    // Third joint column vector of partial derivatives
    j.block<2, 1>(0 ,2) = vector_t{{-sin(a0+a1+a2)}, {cos(a0+a1+a2)}};

    return j;
}

/**
 * Forward kinematics function for a planar
 * 3-DOF manipulator with only revolute links.
 *
 * @param q
 * @return
 */
trafo2d_t forward_kinematics(vector_t const &q) {
    // Check that the joint configuration has the correct size
    assert(q.size() == 3);

    // Define a constant offset between two joints
    trafo2d_t link_offset = trafo2d_t::Identity();
    link_offset.translation()(1) = 1.;

    // Define the start pose
    trafo2d_t trafo = trafo2d_t::Identity();

    for(int joint_idx = 0; joint_idx < 3; joint_idx++) {
        // Add the rotation contributed by this joint
        trafo *= Eigen::Rotation2D<double>(q(joint_idx));

        // Add the link offset to this position
        trafo = trafo * link_offset;
    }

    return trafo;
}

/**
 * Inverse Kinematics function that computes
 * the joint angle configuration required for
 * the 3-DOF manipulator to reach a target EE
 * pose.
 *
 * @param q_start
 * @param goal
 * @return joints angle configuration for given EE target pose
 */
vector_t inverse_kinematics(vector_t const &q_start, trafo2d_t const &goal) {
    // Output joint angles
    vector_t q_output = q_start;

    // Compute the 2D delta between the EE goal and current pose
    vector_t ee_delta = goal.translation() - forward_kinematics(q_output).translation();

    // Apply Newton-Ralphson method
    float alpha = 0.9;
    int iterations = 0;
    while (ee_delta.norm() > 1e-3 && iterations < 200) {
        // Compute the pseudo inverse of the jacobian at configuration
        Eigen::JacobiSVD<matrix_t> svd(jacobian(q_output), Eigen::ComputeThinV | Eigen::ComputeThinU);
        matrix_t J_inv = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixU().transpose();

        // Update joint configuration
        q_output -= J_inv * ee_delta * alpha;

        //TODO: Add angle constraints and other possible soft constraints
        //TODO: Handle angle wrapping
        //TODO: Handle self collisions
        //TODO: Handle singularities

        // Update delta
        ee_delta = goal.translation() - forward_kinematics(q_output).translation();

        // Keep track of iterations
        iterations += 1;
    }

    return q_output;
}

/**
 * An example how the inverse kinematics can be used.
 * It should not be required to change this code.
 */
int main(){
    // Start goal and respective plot
    vector_t q_start(3);
    q_start.setConstant(-0.1);

    // Goal pose
    trafo2d_t goal = trafo2d_t::Identity();
    goal.translation()(0) = 0.5;

    // Perform IK
    vector_t q_ik = inverse_kinematics(q_start, goal);
    std::cout << q_ik << std::endl;

    // Plot start/end angle configurations
    plot_q(q_start, q_ik, "ik");

    return 0;
}