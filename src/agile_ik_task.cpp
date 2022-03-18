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

matrix_t DH(double alfa, double theta, double r, double d)
{

    matrix_t dh(4, 4);

    dh(0, 0) = std::cos(theta);
    dh(0, 1) = -std::sin(theta) * std::cos(alfa);
    dh(0, 2) = std::sin(theta) * std::sin(alfa);
    dh(0, 3) = r * std::cos(theta);

    dh(1, 0) = std::sin(theta);
    dh(1, 1) = std::cos(theta) * std::cos(alfa);
    dh(1, 2) = -std::cos(theta) * std::sin(alfa);
    dh(1, 3) = r * std::sin(theta);

    dh(2, 0) = 0;
    dh(2, 1) = std::sin(alfa);
    dh(2, 2) = std::cos(alfa);
    dh(2, 3) = d;

    dh(3, 0) = 0;
    dh(3, 1) = 0;
    dh(3, 2) = 0;
    dh(3, 3) = 1;

    return dh;
}

///**
// * Function computing the Jacobian
// * matrix of a a planar 3-DOF manipulator.
// *
// * @param q
// * @return 2x3 Jacobian matrix
// */
//matrix_t jacobian(vector_t const &q) {
//    // Jacobian matrix to be populated
//    matrix_t j(2, 3);
//
//    // Prepare angles
//    auto a0 = q(0) * -1;
//    auto a1 = q(1) * -1;
//    auto a2 = q(2) * -1;
//
//    // First joint column vector of partial derivatives
//    j.block<2, 1>(0 ,0) = vector_t{{-sin(a0) - sin(a0+a1) - sin(a0+a1+a2)},
//                                                    {cos(a0) + cos(a0+a1) + cos(a0+a1+a2)}};
//
//    // Second joint column vector of partial derivatives
//    j.block<2, 1>(0 ,1) = vector_t{{-sin(a0+a1) - sin(a0+a1+a2)},
//                                                    {cos(a0+a1) + cos(a0+a1+a2)}};
//
//    // Third joint column vector of partial derivatives
//    j.block<2, 1>(0 ,2) = vector_t{{-sin(a0+a1+a2)},
//                                                    {cos(a0+a1+a2)}};
//
//    //std::cout << j << std::endl;
//
//    return j;
//}

Eigen::MatrixXd jacobian(double theta1, double theta2, double theta3)
{
    double alfa1 = 0;
    double alfa2 = 0;
    double alfa3 = 0;

    double r1 = 1;
    double r2 = 1;
    double r3 = 1;

    double d1 = 0;
    double d2 = 0;
    double d3 = 0;

    matrix_t DH01(4, 4);
    matrix_t DH12(4, 4);
    matrix_t DH23(4, 4);

    DH01 = DH(alfa1, theta1, r1, d1);
    DH12 = DH(alfa2, theta2, r2, d2);
    DH23 = DH(alfa3, theta3, r3, d3);

    matrix_t D01 = DH01;
    matrix_t D01R = D01.block<3, 3>(0, 0);
    matrix_t D01T = D01.block<3, 1>(0, 3);

    matrix_t D02 = DH01 * DH12;
    matrix_t D02R = D02.block<3, 3>(0, 0);
    matrix_t D02T = D02.block<3, 1>(0, 3);

    matrix_t D03 = DH01 * DH12 * DH23;
    matrix_t D03R = D03.block<3, 3>(0, 0);
    matrix_t D03T = D03.block<3, 1>(0, 3);

    Eigen::Vector3d Ri(0, 0, 1);

    Eigen::Vector3d vecD01R = D01R * Ri;
    Eigen::Vector3d vecD02R = D02R * Ri;

    Eigen::Vector3d vecD01T(Eigen::Map<Eigen::Vector3d>(D01T.data(), D01T.cols() * D01T.rows()));
    Eigen::Vector3d vecD02T(Eigen::Map<Eigen::Vector3d>(D02T.data(), D01T.cols() * D02T.rows()));
    Eigen::Vector3d vecD03T(Eigen::Map<Eigen::Vector3d>(D03T.data(), D03T.cols() * D03T.rows()));

    matrix_t J1 = Ri.cross(vecD03T); //R00
    matrix_t J2 = (vecD01R).cross(vecD03T - vecD01T);
    matrix_t J3 = (vecD02R).cross(vecD03T - vecD02T);
    matrix_t J4 = Ri;
    matrix_t J5 = vecD01R;
    matrix_t J6 = vecD02R;

    matrix_t J(2, 3); // we consider only linear velocities

    J(0, 0) = J1(0, 0);
    J(1, 0) = J1(1, 0);
//    J(2, 0) = J1(2, 0);

    J(0, 1) = J2(0, 0);
    J(1, 1) = J2(1, 0);
//    J(2, 1) = J2(2, 0);

    J(0, 2) = J3(0, 0);
    J(1, 2) = J3(1, 0);
//    J(2, 2) = J3(2, 0);

    return J;
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

    std::cout << "Start: " << forward_kinematics(q_output).translation() << std::endl;
    std::cout << "Current output: " << q_output << std::endl;

    // Compute the 2D delta between the EE goal and current pose
    vector_t ee_delta = goal.translation() - forward_kinematics(q_output).translation();

    // Apply Newton-Ralphson method
    uint8_t iterations = 0;
    while (ee_delta.norm() > 1e-3 && iterations < 200) {
        // Displacement
        vector_t displacement = ee_delta * 0.05;

        // Compute the pseudo inverse of the jacobian
        // evaluated at the current joint configuration
//        matrix_t J = jacobian(q_output(0), q_output(1), q_output(2));
//        Eigen::JacobiSVD<matrix_t> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
//        matrix_t J_inv = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixU().transpose();
        matrix_t J = jacobian(q_output(0), q_output(1), q_output(2));
        matrix_t J_inv = J.completeOrthogonalDecomposition().pseudoInverse();

        // Update joint configuration
        //vector_t ee_delta_displacement = 0.001 * ee_delta;
        q_output += J_inv * displacement;

        // If any computed joint angle is outside the range
        // move them back into their defined limits
        if ((q_output(0) > M_PI_2 || q_output(0) < -M_PI_2)) {
            q_output(0) = (M_PI_2 - 0.2) * ((q_output(0) > 0) ? 1 : ((q_output(0) < 0) ? -1 : 0));
        }
        else if ((q_output(1) > M_PI_2 || q_output(1) < -M_PI_2)) {
            q_output(0) = (M_PI_2 - 0.2) * ((q_output(1) > 0) ? 1 : ((q_output(1) < 0) ? -1 : 0));
        }
        else if ((q_output(2) > M_PI_2 || q_output(2) < -M_PI_2)) {
            q_output(0) = (M_PI_2 - 0.2) * ((q_output(2) > 0) ? 1 : ((q_output(2) < 0) ? -1 : 0));
        }

        // Update delta
        ee_delta = goal.translation() - forward_kinematics(q_output).translation();

        std::cout << "Obtained output: " << q_output << std::endl;
        std::cout << "Obtained EE pose: " << forward_kinematics(q_output).translation() << std::endl;
        std::cout << "Delta: " << ee_delta << std::endl;

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
    goal.translation()(0) = 1.;
    goal.translation()(1) = 0.;

    // Perform IK
    vector_t q_ik = inverse_kinematics(q_start, goal);
    std::cout << q_ik << std::endl;

    // Plot start/end angle configurations
    plot_q(q_start, q_ik, "ik");

    return 0;
}