#!/usr/bin/python

#################################################
#                                               #
#   Calder Phillips-Grafflin - WPI/ARC Lab      #
#                                               #
#   Head pointing controller for DRCHubo        #
#                                               #
#   This provides two interfaces:               #
#                                               #
#   1.  PointHeadAction to point the head at a  #
#       given point in the world.               #
#                                               #
#   2.  GeneratePanoramaAction to generate a    #
#       panorama of the robot's environment.    #
#                                               #
#################################################

import rospy
import math
from std_msgs.msg import *
from sensor_msgs.msg import *
import tf
from tf.transformations import *
from transformation_helper import *
import actionlib
from geometry_msgs.msg import *
import hubo_robot_msgs.msg as hrms
import dynamixel_msgs.msg as dmms

class PointHeadController:

    def __init__(self, pan_controller_prefix, tilt_controller_prefix, zero_pan_position, zero_tilt_position, safe_pan_position, safe_tilt_position, target_angular_rate, error_threshold):
        self.zero_pan_position = zero_pan_position
        self.zero_tilt_position = zero_tilt_position
        self.safe_pan_position = safe_pan_position
        self.safe_tilt_position = safe_tilt_position
        self.target_angular_rate = target_angular_rate
        self.error_threshold = error_threshold
        self.valid_pointing_frames = {"Body_NK1":"Body_NK1", "trinocular_base_frame":"trinocular_base_frame", "trinocular_left_sensor_frame":"trinocular_left_sensor_frame", "trinocular_left_optical_frame":"trinocular_left_sensor_frame", "trinocular_center_sensor_frame":"trinocular_center_sensor_frame", "trinocular_center_optical_frame":"trinocular_center_sensor_frame", "trinocular_right_sensor_frame":"trinocular_right_sensor_frame", "trinocular_right_optical_frame":"trinocular_right_sensor_frame", "dual_rgbd_base_frame":"dual_rgbd_base_frame", "rgbd_shortrange_sensor_frame":"rgbd_shortrange_sensor_frame", "rgbd_shortrange_rgb_frame":"rgbd_shortrange_rgb_frame", "rgbd_shortrange_rgb_optical_frame":"rgbd_shortrange_rgb_frame", "rgbd_shortrange_depth_frame":"rgbd_shortrange_depth_frame", "rgbd_shortrange_depth_optical_frame":"rgbd_shortrange_depth_frame", "rgbd_longrange_sensor_frame":"rgbd_longrange_sensor_frame", "rgbd_longrange_rgb_frame":"rgbd_longrange_rgb_frame", "rgbd_longrange_rgb_optical_frame":"rgbd_longrange_rgb_frame", "rgbd_longrange_depth_frame":"rgbd_longrange_depth_frame", "rgbd_longrange_depth_optical_frame":"rgbd_longrange_depth_frame"}
        self.tf_listener = tf.TransformListener()
        self.running = True
        self.last_pan_state = None
        self.last_tilt_state = None
        rospy.loginfo("Configuring PointHeadController...")
        rospy.loginfo("pan_controller_prefix = " + pan_controller_prefix)
        rospy.loginfo("tilt_controller_prefix = " + tilt_controller_prefix)
        self.pan_subscriber = rospy.Subscriber(pan_controller_prefix + "/state", dmms.JointState, self.pan_state_cb)
        self.tilt_subscriber = rospy.Subscriber(tilt_controller_prefix + "/state", dmms.JointState, self.tilt_state_cb)
        rospy.loginfo("Loaded subscribers to head controller states")
        retry_rate = rospy.Rate(1.0)
        while ((self.last_pan_state == None or self.last_tilt_state == None) and not rospy.is_shutdown()):
            rospy.loginfo("Waiting for states from head controllers...")
            retry_rate.sleep()
        rospy.loginfo("...Head controllers ready")
        rospy.loginfo("Loading head controller command publishers...")
        self.pan_command_pub = rospy.Publisher(pan_controller_prefix + "/command", Float64)
        self.tilt_command_pub = rospy.Publisher(tilt_controller_prefix + "/command", Float64)
        rospy.loginfo("...Loaded command publishers")
        rospy.loginfo("...PointHeadController setup finished")

    def pan_state_cb(self, msg):
        self.last_pan_state = msg

    def tilt_state_cb(self, msg):
        self.last_tilt_state = msg

    def BringUp(self):
        rospy.loginfo("Bringing up the PointHeadController to zero state")
        if (self.last_tilt_state.current_pos < -1.1):
            rospy.loginfo("Unstowing head first...")
            unstow_traj = self.BuildTrajectory(self.last_pan_state.current_pos, self.last_tilt_state.current_pos, self.last_pan_state.current_pos, -1.1, self.target_angular_rate)
            result = self.RunTrajectory(unstow_traj)
            if (result):
                rospy.loginfo("Head unstowed")
            else:
                rospy.logerr("HEAD UNABLE TO UNSTOW - will try again")
                return result
        bringup_traj = self.BuildTrajectory(self.last_pan_state.current_pos, self.last_tilt_state.current_pos, self.zero_pan_position, self.zero_tilt_position, self.target_angular_rate)
        result = self.RunTrajectory(bringup_traj)
        if (result):
            rospy.loginfo("Head brought up to the zero position")
        else:
            rospy.logerr("HEAD UNABLE TO ZERO - will try again")
        return result

    def RunActionServer(self):
        self.server = actionlib.SimpleActionServer('drchubo_point_head', hrms.PointHeadAction, self.execute_point_head, False)
        self.server.start()
        safety_rate = rospy.Rate(10.0)
        while not rospy.is_shutdown() and self.running:
            safety_rate.sleep()
        del(self.server)
        return False

    def execute_point_head(self, point_head_goal):
        pan = None
        tilt = None
        if (point_head_goal.target_type == point_head_goal.POINTTARGET):
            [pan, tilt] = self.ComputePointingAngle(point_head_goal)
        elif (point_head_goal.target_type == point_head_goal.ANGLETARGET):
            pan = point_head_goal.pan_angle
            tilt = point_head_goal.tilt_angle
        else:
            rospy.logerr("Aborting PointHeadAction Goal due to invalid target type")
            self.server.set_aborted()
        safe = self.CheckSafetyBounds(pan, tilt)
        if not safe:
            rospy.logerr("Aborting PointHeadAction Goal due to safety violations")
            self.server.set_aborted()
        else:
            point_traj = self.BuildTrajectory(self.last_pan_state.current_pos, self.last_tilt_state.current_pos, pan, tilt, self.target_angular_rate, point_head_goal.min_duration.to_sec())
            result = self.RunTrajectory(point_traj)
            if (result):
                rospy.loginfo("PointHeadAction completed")
                self.server.set_succeeded()
            else:
                rospy.logerr("Unable to complete PointHeadAction")
                self.server.set_aborted()

    def ComputePointingAngle(self, point_head_goal):
        if ("optical" in point_head_goal.pointing_frame and point_head_goal.pointing_axis.z == 0.0):
            rospy.logerr("PointHead specified using an optical frame but pointing axis is not Z")
            return [None, None]
        elif (point_head_goal.pointing_axis.x == 0.0):
            rospy.logerr("PointHead specified using a physical frame but pointing axis is not X")
            return [None, None]
        real_pointing_frame = self.GetRealPointingFrame(point_head_goal.pointing_frame)
        if (real_pointing_frame != None):
            try:
                [tft, tfr] = self.tf_listener.lookupTransform(real_pointing_frame, point_head_goal.target.header.frame_id, rospy.Time())
                target_frame_pose = PoseFromTransform(TransformFromComponents(tft,tfr))
                target_pose = Pose()
                target_pose.orientation.w = 1.0
                target_pose.position.x = point_head_goal.target.point.x
                target_pose.position.y = point_head_goal.target.point.y
                target_pose.position.z = point_head_goal.target.point.z
                pftp = ComposePoses(target_frame_pose, target_pose)
                print "Computed pose of target point:"
                print "X: " + str(pftp.position.x)
                print "Y: " + str(pftp.position.y)
                print "Z: " + str(pftp.position.z)
                target_range = math.sqrt((pftp.position.x ** 2) + (pftp.position.y ** 2) + (pftp.position.z ** 2))
                target_pan = math.atan2(pftp.position.y, pftp.position.x)
                target_tilt = -math.asin(pftp.position.z / target_range)
                print "Computed pan/tilt adjustments:"
                print "Pan change: " + str(target_pan)
                print "Tilt change: " + str(target_tilt)
                real_pan = self.normalize_pan(self.last_pan_state.current_pos + target_pan)
                real_tilt = self.last_tilt_state.current_pos + target_tilt
                print "Computed pan/tilt targets:"
                print "Pan: " + str(real_pan)
                print "Tilt: " + str(real_tilt)
                print "Current pan/tilt:"
                print "Pan: " + str(self.last_pan_state.current_pos)
                print "Tilt: " + str(self.last_tilt_state.current_pos)
                return [real_pan, real_tilt]
            except:
                print "Exception occured trying to compute pointing: ", sys.exc_info()
                rospy.logerr("Unable to compute pointing - this is probably because a frame doesn't exist")
                return [None, None]
        else:
            rospy.logerr("Unable to resolve head pointing frame")
            return [None, None]

    def normalize_pan(self, raw_pan):
        if (raw_pan > (math.pi + self.error_threshold)):
            new_pan = -abs(raw_pan % math.pi)
            return new_pan
        elif (raw_pan < -(math.pi + self.error_threshold)):
            new_pan = abs(raw_pan % math.pi)
            return new_pan
        else:
            return raw_pan

    def GetRealPointingFrame(self, frame_name):
        frame_name = frame_name.lstrip("/")
        try:
            return self.valid_pointing_frames[frame_name]
        except:
            return None

    def CheckSafetyBounds(self, pan, tilt):
        if (pan == None or tilt == None):
            return False
        min_pan = -(math.pi + self.error_threshold)
        max_pan = (math.pi + self.error_threshold)
        min_tilt = -1.1
        max_tilt = math.pi / 4.0
        if (pan > max_pan or pan < min_pan):
            rospy.logerr("Pan value safety violation - requested " + str(pan))
            return False
        if (tilt > max_tilt or tilt < min_tilt):
            rospy.logerr("Tilt value safety violation - requested " + str(tilt))
            return False
        return True

    def BringDown(self):
        rospy.loginfo("Bringing up the PointHeadController to zero state")
        bringdown_traj = self.BuildTrajectory(self.last_pan_state.current_pos, self.last_tilt_state.current_pos, self.safe_pan_position, self.safe_tilt_position, self.target_angular_rate)
        result = self.RunTrajectory(bringdown_traj)
        if (result):
            rospy.loginfo("Head brought down to the safe position")
        else:
            rospy.logerr("HEAD UNABLE TO ZERO - will try again")
            return result
        rospy.loginfo("Tucking head into stow position")
        stow_traj = self.BuildTrajectory(self.last_pan_state.current_pos, self.last_tilt_state.current_pos, self.safe_pan_position, self.safe_tilt_position - 0.1, self.target_angular_rate)
        result = self.RunTrajectory(stow_traj)
        if (result):
            rospy.loginfo("Head stowed")
        else:
            rospy.logerr("HEAD UNABLE TO STOW - will not try again")
        return True

    def BuildTrajectory(self, start_pan, start_tilt, end_pan, end_tilt, angular_rate, execution_time=None):
        steps_per_sec = 50.0
        steptime = 1.0 / steps_per_sec
        trajectory_states = []
        angle = self.compute_distance(start_pan, start_tilt, end_pan, end_tilt)
        nominal_time = angle / angular_rate
        if (execution_time != None):
            if ((execution_time / angle) > angular_rate):
                rospy.logwarn("Desired execution time would require exceeding velocity bounds")
            else:
                nominal_time = execution_time
        steps = int(math.ceil(nominal_time * steps_per_sec))
        if (steps == 1):
            return [[end_pan, end_tilt, steptime]]
        elif (steps < 1):
            return []
        else:
            for step in range(steps):
                p = float(step) / float(steps - 1)
                pan = ((end_pan - start_pan) * p) + start_pan
                tilt = ((end_tilt - start_tilt) * p) + start_tilt
                trajectory_states.append([pan, tilt, steptime])
            return trajectory_states

    def compute_distance(self, start_pan, start_tilt, end_pan, end_tilt):
        pan_dist = abs(end_pan - start_pan)
        tilt_dist = abs(end_tilt - start_tilt)
        real_dist = math.sqrt((pan_dist ** 2) + (tilt_dist ** 2))
        return real_dist

    def RunTrajectory(self, trajectory, debug=False):
        if (debug):
            rospy.logwarn("Running in debug mode, trajectory will not execute")
            for state in trajectory:
                print state
            return False
        rospy.loginfo("Executing head trajectory...")
        for [pan, tilt, timestep] in trajectory:
            self.pan_command_pub.publish(pan)
            self.tilt_command_pub.publish(tilt)
            rospy.sleep(timestep)
        if (abs(self.last_pan_state.error) > self.error_threshold or abs(self.last_tilt_state.error) > self.error_threshold):
            self.pan_command_pub.publish(self.last_pan_state.current_pos)
            self.tilt_command_pub.publish(self.last_tilt_state.current_pos)
            return False
        else:
            return True

if __name__ == '__main__':
    rospy.init_node("drchubo_head_controller")
    rospy.loginfo("Loading PointHeadController...")
    pan_controller_prefix = rospy.get_param("~pan_controller_prefix", "/head_pan_controller")
    tilt_controller_prefix = rospy.get_param("~tilt_controller_prefix", "/head_tilt_controller")
    zero_pan_position = rospy.get_param("~zero_pan_position", 0.0)
    zero_tilt_position = rospy.get_param("~zero_tilt_position", 0.0)
    safe_pan_position = rospy.get_param("~safe_pan_position", 0.0)
    safe_tilt_position = rospy.get_param("~safe_tilt_position", -1.1)
    target_angular_rate = rospy.get_param("~target_angular_rate", (math.pi / 4.0))
    error_threshold = rospy.get_param("~error_threshold", (math.pi / 36.0))
    PHC = PointHeadController(pan_controller_prefix, tilt_controller_prefix, zero_pan_position, zero_tilt_position, safe_pan_position, safe_tilt_position, target_angular_rate, error_threshold)
    retry_rate = rospy.Rate(1.0)
    result = False
    while not result and not rospy.is_shutdown():
        result = PHC.BringUp()
        retry_rate.sleep()
    result = False
    rospy.loginfo("Starting PointHeadAction server...")
    result = PHC.RunActionServer()
    if (not result):
        rospy.logwarn("PointHeadController actionserver stopped, attempting to safely shutdown")
    result = False
    while not result and not rospy.is_shutdown():
        result = PHC.BringDown()
        retry_rate.sleep()
    rospy.loginfo("...Operations complete, PointHeadController safed and shutdown")
