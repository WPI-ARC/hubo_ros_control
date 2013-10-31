/*
 * Copyright (c) 2013, Drexel DARPA Robotics Challenge team
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/* Author: Jim Mainprice (WPI) */

// ROS & includes
#include <ros/ros.h>
// Boost includes
#include <boost/thread.hpp>
// Message and action includes for Hubo actions
#include <hubo_robot_msgs/JointTrajectory.h>
#include <hubo_robot_msgs/JointTrajectoryState.h>
#include <hubo_robot_msgs/JointTrajectoryPoint.h>
#include <hubo_robot_msgs/ComplianceConfig.h>
#include <rosgraph_msgs/Clock.h>
// Includes for ACH and hubo-motion-rt
#include <Hubo_Control.h>
// Trajectory structure
#include <hubo_trajectory_interface/hubo_trajectory.hpp>
// Boost includes
#include <boost/thread.hpp>

class HuboMotionRtController
{
public:
    HuboMotionRtController( ros::NodeHandle &n );
    ~HuboMotionRtController();

    void shut_down();

    //! This runs in a second thread in the background.
    //! The thread grabs the latest states and setpoints from hubo's joints
    //! and repacks them into ROS messages sent to the trajectory action server.
    void publish_loop();

    //! Given a string name of a joint, it looks it up in the list of joint names
    //! to determine the joint index used in hubo-ach. If the name can't be found, it returns -1.
    int ach_index_lookup( const std::string& joint_name );

    //! Sets the ach finger joints ids
    void set_ach_finger_joints_ids();

    //! Returns true if joint id is finger
    bool is_ach_finger_joints_id(int jnt);

    //! Callback when the ROS trajectory is received.
    //! Sets the compliance mode, proceeds to hubo-ach joint mapping,
    virtual void trajectory_cb( const hubo_robot_msgs::JointTrajectory& ros_traj );

    void tsnorm(struct timespec *ts);

    //! Gets the current time from hubo-ach
    double get_time();

    //! Gets the elapsed time from a reference time
    double time_from_ref( const double& t_ref );

    //! Set joint compliance ON, the coefficient are defiended in the trajectory
    void set_arms_compliance_on();

    //! Set arms compliance OFF
    void set_arms_compliance_off();

    //! Set nominal velocity
    void set_nominal_vel_and_acc();

    //! Executes a peicewise linear trajectory
    //! way points are linearly interpolated, sends values at 200Hz through ach
    virtual bool execute_linear_trajectory();

    //! Spins the node and lauches when a trajectory is received
    virtual void main_loop();

protected:
    ros::NodeHandle node_;
    ros::NodeHandle nhp_;

    // Publisher and subscriber
    ros::Publisher state_pub_;
    ros::Publisher clock_pub_;
    ros::Subscriber traj_sub_;

    struct timespec* ts_;

    // Thread for listening to the hubo state and republishing it
    boost::thread* pub_thread_;
    boost::thread* traj_thread_;

    double publish_average_periode_;
    double commands_average_periode_;

    // Joint name and mapping storage
    std::vector<std::string> joint_names_;
    std::map<std::string,int> joint_mapping_;
    std::vector<int> finger_joints_;
    std::vector<int> all_joints_;
    std::vector<int> active_joints_;
    std::vector<double> error_;

    // Compliance
    std::vector<bool> active_joints_compliance_;
    std::vector<std::string> compliant_joint_names_;
    std::vector<double> compliance_kp_;
    std::vector<double> compliance_kd_;

    Hubo_Control* hubo_;
    Hubo::Trajectory hubo_traj_;
    ach_channel_t chan_hubo_ctrl_state_pub_;

    // Execution state
    bool running_;
};
