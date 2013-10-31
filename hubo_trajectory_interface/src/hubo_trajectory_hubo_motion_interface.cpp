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

// Trajectory structure
#include <hubo_trajectory_interface/hubo_trajectory_hubo_motion_interface.hpp>

// for RT
#include <stdlib.h>
#include <sys/mman.h>

#include <vector>
// System includes to handle safe shutdown
#include <signal.h>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

using std::cout;
using std::endl;

/*
 * Node must be run with root privileges (sudo su and edit the bashrc of root)
 * from their roslaunch the action server and this node
 * We should add cancelation of the trajectory execution at this point
 *
 * These values may need to be experimentally determined, and the SPIN_RATE parameter
 * may need to dynamically change based on the timings on the incoming trajectory.
*/

static bool sim_mode = false;

// Timing info
#define NSEC_PER_SEC    1000000000

double COEF_SIM_TIME = 1; // 1 for normal mode

double SPIN_RATE = 3.0; // Rate in hertz at which to send trajectory chunks

// static int interval = 5000000; // 200 hz (0.005 sec)
static int interval = 20000000; // 50 hz (0.02 sec)

#define MAX_SAFE_STACK (1024*1024)
// The maximum stack size which is
// guaranteed safe to access without faulting

void stack_prefault(void) {
    unsigned char dummy[MAX_SAFE_STACK];
    memset( dummy, 0, MAX_SAFE_STACK );
}

HuboMotionRtController::HuboMotionRtController( ros::NodeHandle &n ) : node_(n), nhp_("~")
{
    //Get simulation flag
    nhp_.param( "/use_sim_time", sim_mode, false );
    if( sim_mode )
    {
        ROS_INFO("Using simulation mode");
        COEF_SIM_TIME = 20;
        SPIN_RATE *= COEF_SIM_TIME;
    }
    running_;
    hubo_=NULL;

    ROS_INFO( "Attempting to start JointTrajectoryAction controller interface..." );

    // Gets all the active joint names
    XmlRpc::XmlRpcValue joint_names;
    if (!nhp_.getParam("joints", joint_names))
    {
        ROS_FATAL("No joints given. (namespace: %s)", nhp_.getNamespace().c_str());
        exit(1);
    }
    if (joint_names.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
        ROS_FATAL("Malformed joint specification.  (namespace: %s)", nhp_.getNamespace().c_str());
        exit(1);
    }
    for ( int i=0; i<int(joint_names.size()); i++ )
    {
        XmlRpc::XmlRpcValue &name_value = joint_names[i];
        if (name_value.getType() != XmlRpc::XmlRpcValue::TypeString)
        {
            ROS_FATAL("Array of joint names should contain all strings.  (namespace: %s)", nhp_.getNamespace().c_str());
            exit(1);
        }
        joint_names_.push_back( std::string(name_value) );
    }

    all_joints_.clear();

    // Gets the hubo ach index for each joint
    for ( int i=0; i<int(joint_names_.size()); i++ )
    {
        std::string ns = std::string("mapping/") + joint_names_[i];
        int h;
        nhp_.param( ns + "/huboachid", h, -1);
        joint_mapping_[joint_names_[i]] = h;
        all_joints_.push_back( h );
        //ROS_INFO( "joint_names_[%d] : %s\n", i, joint_names_[i].c_str() );
    }

    // Stores the ach finger joint ids
    set_ach_finger_joints_ids();

    // Stores the active joints
    for( size_t i=0;i<all_joints_.size();i++)
    {
        if( !is_ach_finger_joints_id( all_joints_[i] ) )
            active_joints_.push_back( all_joints_[i] );
    }

    // Sets up state publisher
    std::string pub_path = node_.getNamespace() + "/state";
    state_pub_ = node_.advertise<hubo_robot_msgs::JointTrajectoryState>(pub_path, 1);

    // Sets up the trajectory subscriber
    std::string sub_path = node_.getNamespace() + "/command";
    traj_sub_ = node_.subscribe( sub_path, 1, &HuboMotionRtController::trajectory_cb, this );
    ROS_INFO("Loaded trajectory interface to hubo-motion-rt");

    running_ = false;

    // Sets up Hubo Control daemon (hubo-motion-rt)
    hubo_ = new Hubo_Control();
    ROS_INFO("Hubo_Control has started!!!!");

    if( !sim_mode )
    {
        // Declare ourself as a real time task
        struct sched_param param;
        param.sched_priority = 49; // 49 is higest priority
        if(sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
            ROS_FATAL("sched_setscheduler failed");
            exit(1);
        }

        // Lock memory
        if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1) {
            ROS_FATAL("mlockall failed");
            exit(1);
        }

        // Pre-fault our stack
        stack_prefault();
    }

    // Sets up the thread for getting data from hubo and publishing it
    pub_thread_ = new boost::thread( &HuboMotionRtController::publish_loop, this );

    // Starts main loop
    main_loop();
}

HuboMotionRtController::~HuboMotionRtController()
{
    shut_down();
}

void HuboMotionRtController::main_loop()
{
    // Spins until killed
    while (ros::ok())
    {
        //ROS_INFO("loop in robot-motion-rt controller");
        if( !running_ && !hubo_traj_.empty() )
        {
            running_ = true;
            // starts the thread for trajectory execution
            traj_thread_ = new boost::thread( &HuboMotionRtController::execute_linear_trajectory, this );
        }

        // Wait long enough before sending the next one and receiving the running signal
        ros::spinOnce();
        ros::Rate looprate( SPIN_RATE );
        looprate.sleep();
    }
}

void HuboMotionRtController::shut_down()
{
    hubo_traj_.clear();
    ROS_INFO("Starting safe shutdown...");

    //pub_thread->join();
    ROS_INFO("All threads done");

    ach_close(&chan_hubo_ctrl_state_pub_);
    ROS_INFO("ach channels closed shutting down!");

    ros::shutdown();
}

//! This runs in a second thread in the background.
//! The thread grabs the latest states and setpoints from hubo's joints
//! and repacks them into ROS messages sent to the trajectory action server.
void HuboMotionRtController::publish_loop()
{
    ROS_INFO("publishLoop");

    ach_status_t r;
    r = ach_open( &chan_hubo_ctrl_state_pub_,  CTRL_CHAN_STATE, NULL );
    if (r != ACH_OK)
    {
        ROS_FATAL("Could not open ACH channel: CTRL_CHAN_STATE !");
        exit(1);
    }

    hubo_ctrl_state_t H_ctrl_state;
    memset(&H_ctrl_state, 0, sizeof(H_ctrl_state));

    // Initialize time
    int nb_loops = 0;
    publish_average_periode_ = 0.0;
    double t_last = get_time();
    double t_total = 0.0;

    struct timespec t;
    clock_gettime( 0,&t);

    // Publish the latest hubo state back out
    hubo_robot_msgs::JointTrajectoryState cur_state;
    // Set the names
    cur_state.joint_names = joint_names_;
    size_t num_joints = cur_state.joint_names.size();
    // Make the empty states
    hubo_robot_msgs::JointTrajectoryPoint cur_setpoint;
    hubo_robot_msgs::JointTrajectoryPoint cur_actual;
    hubo_robot_msgs::JointTrajectoryPoint cur_error;
    // Resize the states
    cur_setpoint.positions.resize(num_joints);
    cur_setpoint.velocities.resize(num_joints);
    cur_setpoint.accelerations.resize(num_joints);
    cur_actual.positions.resize(num_joints);
    cur_actual.velocities.resize(num_joints);
    cur_actual.accelerations.resize(num_joints);
    cur_error.positions.resize(num_joints);
    cur_error.velocities.resize(num_joints);
    cur_error.accelerations.resize(num_joints);

    // Loop until node shutdown
    while (ros::ok())
    {
        clock_nanosleep(0,TIMER_ABSTIME,&t, NULL);
        
        size_t fs;

        // Get latest state from HUBO-MOTION (this is used to populate the desired values)
        ach_status_t r = ach_get( &chan_hubo_ctrl_state_pub_,  &H_ctrl_state, sizeof(H_ctrl_state), &fs, NULL, ACH_O_LAST );

        if( r != ACH_OK && r != ACH_MISSED_FRAME && r != ACH_STALE_FRAMES )
        {
            ROS_ERROR("get ach channel for H_ctrl_state failed in [publishing loop] : %s" , ach_result_to_string(r) );
            //continue;
        }
        else if (fs != sizeof(H_ctrl_state))
        {
            //ROS_ERROR("Hubo ref size error! [publishing loop] with ach channel %s" , ach_result_to_string(r));
            continue;
        }

        cur_state.header.stamp = ros::Time::now();

        // Fill in the setpoint and actual & calc the error_ in the process
        for (size_t i=0; i<num_joints; i++)
        {
            // Fill in the setpoint and actual data
            // Values that we don't have data for are set to NAN
            int hubo_index = ach_index_lookup( cur_state.joint_names[i] );
            if (hubo_index >= 0)
            {
                cur_setpoint.positions[i]       = H_ctrl_state.requested_pos[hubo_index];
                cur_setpoint.velocities[i]      = H_ctrl_state.requested_vel[hubo_index];
                cur_setpoint.accelerations[i]   = H_ctrl_state.requested_acc[hubo_index];

                cur_actual.positions[i]         = H_ctrl_state.actual_pos[hubo_index];
                cur_actual.velocities[i]        = H_ctrl_state.actual_vel[hubo_index];
                cur_actual.accelerations[i]     = H_ctrl_state.actual_acc[hubo_index];
                // Calc the error
                cur_error.positions[i] =        cur_setpoint.positions[i] -     cur_actual.positions[i];
                cur_error.velocities[i] =       cur_setpoint.velocities[i] -    cur_actual.velocities[i];
                cur_error.accelerations[i] =    cur_setpoint.accelerations[i] - cur_actual.accelerations[i];
            }
        }

        // Publish State (pack them together)
        cur_state.desired = cur_setpoint;
        cur_state.actual = cur_actual;
        cur_state.error = cur_error;
        state_pub_.publish( cur_state );

        // Publish Time
        double t_end = get_time();
        //ROS_INFO("Time : %f",t_end);
	
        // Compute average publishing periode
        t_total += (t_end - t_last); // add periode to total time
        publish_average_periode_ = t_total / double(++nb_loops);
        t_last = t_end;

        // Waits to attain user specified frequency
        t.tv_nsec+=interval;
        tsnorm(&t);
    }
}

//! Given a string name of a joint, it looks it up in the list of joint names
//! to determine the joint index used in hubo-ach. If the name can't be found, it returns -1.
int HuboMotionRtController::ach_index_lookup( const std::string& joint_name )
{
    for ( size_t i=0; i<joint_names_.size(); i++ )
    {
        if( joint_names_[i] == joint_name )
        {
            return joint_mapping_[joint_names_[i]];
        }
    }
    return -1;
}

//! Sets the ach finger joints
//! Remove them from the active set
//! Warning hard coded joint names (better that joint ids...)
void HuboMotionRtController::set_ach_finger_joints_ids()
{
    finger_joints_.clear();
    finger_joints_.push_back( ach_index_lookup( "RF1" ) );
    finger_joints_.push_back( ach_index_lookup( "RF2" ) );
    finger_joints_.push_back( ach_index_lookup( "LF1" ) );

    ROS_INFO("set finger joints (%d)",int(finger_joints_.size()));
}

//! Returns true if it is a finger joint
bool HuboMotionRtController::is_ach_finger_joints_id(int jnt)
{
    for ( size_t i=0; i<finger_joints_.size(); i++ )
    {
        if( jnt == finger_joints_[i] )
        {
            return true;
        }
    }
    return false;
}

//! Callback when the ROS trajectory is received.
//! Sets the compliance mode, proceeds to hubo-ach joint mapping,
void HuboMotionRtController::trajectory_cb( const hubo_robot_msgs::JointTrajectory& ros_traj )
{
    ROS_INFO("Trajectory received");

    // Before we do anything, check if the trajectory is empty
    // this is a special "stop" value that flushes the current stored trajectory
    if ( ros_traj.points.empty() )
    {
        ROS_INFO("Flushing current trajectory");
        return;
    }

    ROS_INFO("Reprocessing trajectory with %ld elements into chunks", ros_traj.points.size());

    // Set trajectory
    hubo_traj_.clear();

    // Set compliance
    compliant_joint_names_ = ros_traj.compliance.joint_names;
    if( !compliant_joint_names_.empty() )
    {
        compliance_kp_ = ros_traj.compliance.compliance_kp;
        compliance_kd_ = ros_traj.compliance.compliance_kd;
    }

    ros::Duration base_time(0.0);

    // Get hubo current config
    hubo_->update(true);
    Hubo::Milestone q_cur; // Hubo::Milestone is defined in hubo_trajectory.hpp
    q_cur.second.resize( HUBO_JOINT_COUNT );
    for ( size_t i=0; i<q_cur.second.size(); i++)
        q_cur.second[i] = hubo_->getJointAngleState( i );

    // Process all points
    for ( size_t i=0; i<ros_traj.points.size();i++ )
    {
        // Make sure the JointTrajectoryPoint gets retimed to match its new trajectory chunk
        hubo_robot_msgs::JointTrajectoryPoint cur_point = ros_traj.points[i];

        // Retiming based on the receiving times
        cur_point.time_from_start = cur_point.time_from_start - base_time;

        // Make sure position, velocity, and acceleration are all the same length
        int size = cur_point.positions.size();
        cur_point.velocities.resize(size);
        cur_point.accelerations.resize(size);

        // Store the point as in linear peicewise trajectory
        Hubo::Milestone q;
        q.first = cur_point.time_from_start.toSec();
        q.second = q_cur.second;

        // Now, overwrite with the commands in the current trajectory
        for ( size_t j=0; j<cur_point.positions.size(); j++)
        {
            int index = ach_index_lookup( ros_traj.joint_names[j] );
            if (index != -1)
            {
                q.second[index] = cur_point.positions[j];
                // TODO have a mode for handleing velocity and acceleration profiles
                //processed.velocities[index] = cur_point.velocities[i];
                //processed.accelerations[index] = cur_point.accelerations[i];
            }
        }

        hubo_traj_.push_back( q );
    }

    ROS_INFO("Received a new trajectory with %ld elements", ros_traj.points.size());
}

//! compute t
void HuboMotionRtController::tsnorm( struct timespec* t )
{
    // calculates the next shot
    while (t->tv_nsec >= NSEC_PER_SEC) {
        t->tv_nsec -= NSEC_PER_SEC;
        t->tv_sec++;
    }
}

//! Gets the current time from hubo-ach
double HuboMotionRtController::get_time()
{
    if( sim_mode )
    {
        hubo_->update(true);
        double time = hubo_->getTime();
//        cout << "time : " << time << endl;
        return time;
    }
    else
    {
        timeval tim;
        gettimeofday(&tim, NULL);
        double tu=tim.tv_sec+(tim.tv_usec/1000000.0);
        return tu;
    }
}

//! Gets the elapsed time from a reference time
double HuboMotionRtController::time_from_ref( const double& t_ref )
{
    return get_time() - t_ref;
}

//! Set joint compliance ON, the coefficient are defiended in the trajectory
void HuboMotionRtController::set_arms_compliance_on()
{
    for( size_t i=0;i<compliant_joint_names_.size(); i++ )
    {
        int index = ach_index_lookup( compliant_joint_names_[i] );
        if (index != -1)
        {
            //hubo_->setJointCompliance( index, ON );
            //            hubo_->setJointCompliance( index, ON, compliance_kp_[i], compliance_kd_[i] );
            //            cout << "compliance_kp_[" << i << "] : " << compliance_kp_[i] << endl;
            //            cout << "compliance_kd_[" << i << "] : " << compliance_kp_[i] << endl;
        }
    }
}

//! Set arms compliance OFF
void HuboMotionRtController::set_arms_compliance_off()
{
    //hubo_->setArmCompliance( RIGHT, OFF );
    //hubo_->setArmCompliance( LEFT,  OFF );
}

//! Set nominal velocity
void HuboMotionRtController::set_nominal_vel_and_acc()
{
    ArmVector rArmSpeedDef, rLegSpeedDef, rArmAccDef, rLegAccDef;
    LegVector lArmSpeedDef, lLegSpeedDef, lArmAccDef, lLegAccDef;

    hubo_->getArmNomSpeeds( RIGHT, rArmSpeedDef );
    hubo_->getArmNomSpeeds( LEFT,  lArmSpeedDef );
    hubo_->getLegNomSpeeds( RIGHT, rLegSpeedDef );
    hubo_->getLegNomSpeeds( LEFT,   lLegSpeedDef );

    hubo_->getArmNomAcc( RIGHT, rArmAccDef );
    hubo_->getArmNomAcc( LEFT,  lArmAccDef );
    hubo_->getLegNomAcc( RIGHT, rLegAccDef );
    hubo_->getLegNomAcc( LEFT,  lLegAccDef );
}

//! Executes a peicewise linear trajectory
//! way points are linearly interpolated, sends values at 200Hz through ach
bool HuboMotionRtController::execute_linear_trajectory()
{
    if( hubo_ == NULL )
    {
        ROS_WARN("hubo object not initilized");
        return 0;
    }

    ROS_INFO("Load trajectory ---------------------- ");
    ROS_INFO("  time length : %f", hubo_traj_.get_length());
    ROS_INFO("  time nb of milestones : %d", hubo_traj_.get_number_of_milestones());

    set_nominal_vel_and_acc();

//    const double dt = 0.005; // 200Hz
//    const double dt = 0.02; // 50Hz
    const double dt = double(interval) * 1e-9; // 50Hz
    double t_start = get_time();
    double t_length = hubo_traj_.get_length();
    double t_cur = time_from_ref( t_start );
    double t_end = 0.0;
    double t_total= 0.0;
    int nb_loops = 0;
    int over_shoot = 0;
    commands_average_periode_ = 0.0;

    struct timespec t;
    clock_gettime( 0, &t );

    for(int i=0; i<int(active_joints_.size()); i++)
    {
        int jnt = active_joints_[i];
        hubo_->setJointTrajCorrectness( jnt, 0.05 );
    }

    // Set the compliance flag for each joint
    active_joints_compliance_.resize(active_joints_.size());
    for(size_t i=0; i<int(active_joints_.size()); i++)
    {
        bool enable_joint_compliance = false;
        int real_joint_index = active_joints_[i];
        for (size_t j=0; j<compliant_joint_names_.size(); j++)
        {
            std::string& name = compliant_joint_names_[j];
            int name_joint_index = ach_index_lookup( name );
            if (real_joint_index == name_joint_index)
            {
                enable_joint_compliance = true;
            }
        }
        active_joints_compliance_[i] = enable_joint_compliance;
    }

    while( t_cur <= t_length )
    {
        if( !sim_mode )
            clock_nanosleep(0,TIMER_ABSTIME,&t, NULL);

        // TODO implement a debugging tool (statistics on over shoot)
        Hubo::Vector q_t0 = hubo_traj_.get_config_at_time( t_cur );
        Hubo::Vector q_t1 = hubo_traj_.get_config_at_time( t_cur + dt );

        error_.resize( HUBO_JOINT_COUNT, 0.0 );

        // Treat body joints
        for(size_t i=0; i<int(active_joints_.size()); i++)
        {
            int jnt = active_joints_[i];
            bool enable_compliance = active_joints_compliance_[i];
            // TODO test with smoothing (traj mode) 50 Hz
            hubo_->setJointTraj( jnt, q_t0[jnt], (q_t1[jnt]-q_t0[jnt])/dt, false );
            hubo_->setCompliance( jnt, enable_compliance );
            error_[jnt] = q_t0[jnt] - hubo_->getJointAngleState( jnt );
        }

        // Treat finger joints
        for(size_t i=0; i<int(finger_joints_.size()); i++)
        {
            int jnt = finger_joints_[i];
            hubo_->passJointAngle( jnt, q_t0[jnt] );
        }

        hubo_->sendControls();

        t_end = time_from_ref( t_start );

        if( sim_mode )
        {
            while( (t_end - t_cur) <= dt )
            {
                t_end = time_from_ref( t_start );
                usleep(100);
            }
        }

        // Computes average publishing periode
        t_total += (t_end - t_cur); // add periode to total time
        commands_average_periode_ = t_total / double(++nb_loops);
        t_cur = t_end;

        if( !sim_mode )
        {
            // Waits to attain user specified frequency
            t.tv_nsec+=interval;
            tsnorm(&t);
        }
    }

    ROS_INFO( "Total time : %f", t_total );
    ROS_INFO( "Nb Loops : %d", nb_loops );
    ROS_INFO( "Command rate : %f sec, (%f Hz)", commands_average_periode_, double(1)/commands_average_periode_ );

    double pub_average = publish_average_periode_;
    ROS_INFO( "Publishing rate : %f sec, (%f Hz)", pub_average, double(1)/pub_average );

    hubo_traj_.clear();
    set_nominal_vel_and_acc();
    running_ = false;
    return true;
}

//! This needs to be global so the signal handler can use it
HuboMotionRtController* g_hmrc;

//! Signal handler to catch SIGINT (the shutdown command) and attempt to safely shutdown the trajectory interface
void shutdown(int signum)
{
    ROS_WARN("Attempting to shutdown node...");
    if (g_hmrc != NULL)
    {
        delete g_hmrc;
    }
    else
    {
        ROS_WARN("HJTAS not yet loaded, aborting the load process and shutting down");
    }
    ros::shutdown();
}

//! Main function that spins the ros node
int main(int argc, char** argv)
{
    ROS_INFO("Starting HuboMotionRtContorler action server node...");
    ros::init( argc, argv, "hubo_joint_trajectory_hubo_motion_interface_node", ros::init_options::NoSigintHandler);
    ros::NodeHandle node;
    // Register a signal handler to safely shutdown the node
    signal(SIGINT, shutdown);
    ROS_INFO("Attempting to start HuboMotionRtContorler action server...");
    g_hmrc = new HuboMotionRtController( node );
    return 0;
}
