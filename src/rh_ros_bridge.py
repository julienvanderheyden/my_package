#!/usr/bin/env python3
import rospy
import csv
import os
import argparse
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ShadowHandBridge:
    def __init__(self, store_data, csv_filename, row_limit):

        rospy.init_node('shadowhand_ros_bridge', anonymous=True)

        ### STATE LISTENER : Listens to ShadowHand joint values, re-order them and post it on the right state topic ###

        # Joint order
        self.joint_names_ordered = ["rh_WRJ1", "rh_WRJ2", "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4", "rh_MFJ1",
                                    "rh_MFJ2", "rh_MFJ3", "rh_MFJ4", "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4", "rh_LFJ1",
                                    "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5", "rh_THJ1", "rh_THJ2", "rh_THJ3",
                                    "rh_THJ4", "rh_THJ5"]

        # Publisher and subscriber
        self.state_topic = "/shadowhand_state_topic"
        self.state_pub = rospy.Publisher(self.state_topic, JointState, queue_size=10)

        rospy.Subscriber("/joint_states", JointState, self.state_callback)

        rospy.loginfo(f"Listening to /joint_states and publishing reordered positions on {self.state_topic}")

        ### TOPIC SPLITTER : Split the command sent by Julia to the correct joint command topics ###

        # Publisher and subscriber
        self.julia_command_topic = "/shadowhand_command_topic"
        self.joints_command_topics = ["/sh_rh_wrj1_position_controller/command", "/sh_rh_wrj2_position_controller/command",
                           "/sh_rh_ffj0_position_controller/command", "/sh_rh_ffj3_position_controller/command", "/sh_rh_ffj4_position_controller/command",
                           "/sh_rh_mfj0_position_controller/command", "/sh_rh_mfj3_position_controller/command", "/sh_rh_mfj4_position_controller/command",
                           "/sh_rh_rfj0_position_controller/command", "/sh_rh_rfj3_position_controller/command", "/sh_rh_rfj4_position_controller/command",
                           "/sh_rh_lfj0_position_controller/command", "/sh_rh_lfj3_position_controller/command", "/sh_rh_lfj4_position_controller/command",
                           "/sh_rh_lfj5_position_controller/command", "/sh_rh_thj1_position_controller/command", "/sh_rh_thj2_position_controller/command",
                           "/sh_rh_thj3_position_controller/command", "/sh_rh_thj4_position_controller/command", "/sh_rh_thj5_position_controller/command"]

        self.joint_command_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.joints_command_topics]
        rospy.Subscriber(self.julia_command_topic, Float64MultiArray, self.command_callback)

        # Joint limits and coupling
        self.joints_limits = [
                               [-0.69, 0.48], [-0.52, 0.71],
                               [0,1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34], [0, 0.78],
                               [-0.26, 1.57], [-0.69, 0.69], [-0.2, 0.2], [0, 1.22], [-1.04, 1.04]
                            ]

        self.coupled_fingers = {2, 5, 8, 11}

        # Logging rate
        self.rate = rospy.Rate(2)  # Print state 2 times per second
        self.joints_names = ["WRJ1", "WRJ2", "FFJ1", "FFJ2", "FFJ3", "FFJ4", "MFJ1", "MFJ2", "MFJ3", "MFJ4", "RFJ1", "RFJ2", "RFJ3", "RFJ4",
                            "LFJ1", "LFJ2", "LFJ3", "LFJ4", "LFJ5", "THJ1", "THJ2", "THJ3", "THJ4", "THJ5"]
        self.joint_states = [0.0] * len(self.joints_names)

        rospy.loginfo(f"Subscribed to: {self.julia_command_topic}")
        rospy.loginfo(f"Publishing to: {self.joints_command_topics}")

        ### DIAGNOSTICS LISTENER : Listens to /diagnostics and buffers per-motor data ###
        #
        # The /diagnostics topic publishes DiagnosticArray messages at ~2 Hz. Each status entry
        # whose name matches "rh SRDMotor <JOINT>" (e.g. "rh SRDMotor WRJ1") carries a flat list
        # of key/value pairs. We extract three fields per motor:
        #
        #   "Measured Current"      -> diag_current
        #   "Measured Effort"       -> diag_effort
        #   "Last Commanded Effort" -> diag_cmd_effort
        #
        # Some joints have no physical motor (FFJ1, FFJ2 …). Their status entry is present but
        # has an empty values list; we silently skip them and leave their buffer slots as "".
        #
        # The buffer (self.latest_diag) is updated by diagnostics_callback() whenever a new
        # DiagnosticArray arrives. It is consumed (and cleared) once per CSV row inside
        # state_callback(), so each row carries the freshest available diagnostics snapshot
        # without ever leaving a row empty.

        # Motor names as they appear in the "name" field of DiagnosticArray status entries.
        # This list mirrors self.joints_names: index i in joints_names <-> index i here.
        self.diag_motor_names = [
            "rh SRDMotor WRJ1", "rh SRDMotor WRJ2",
            "rh SRDMotor FFJ0", "rh SRDMotor FFJ3", "rh SRDMotor FFJ4",
            "rh SRDMotor MFJ0", "rh SRDMotor MFJ3", "rh SRDMotor MFJ4",
            "rh SRDMotor RFJ0", "rh SRDMotor RFJ3", "rh SRDMotor RFJ4",
            "rh SRDMotor LFJ0",  "rh SRDMotor LFJ3", "rh SRDMotor LFJ4", "rh SRDMotor LFJ5",
            "rh SRDMotor THJ1", "rh SRDMotor THJ2", "rh SRDMotor THJ3", "rh SRDMotor THJ4", "rh SRDMotor THJ5",
        ]

        self.diagnostics_joints = ["WRJ1", "WRJ2", "FFJ0", "FFJ3", "FFJ4", "MFJ0", "MFJ3", "MFJ4", "RFJ0", "RFJ3", "RFJ4",
                                   "LFJ0", "LFJ3", "LFJ4", "LFJ5", "THJ1", "THJ2", "THJ3", "THJ4", "THJ5"]

        # Buffer: holds the latest diagnostics values for every motor.
        # Each entry is a dict with keys "current", "effort", "cmd_effort", or "" when unavailable.
        n = len(self.diag_motor_names)
        self.latest_diag = [{"current": "", "effort": "", "cmd_effort": ""} for _ in range(n)]
        # Flag: True once at least one diagnostics message has been received.
        self.diag_received = False

        rospy.Subscriber("/diagnostics", DiagnosticArray, self.diagnostics_callback)
        rospy.loginfo("Subscribed to /diagnostics for per-motor current / effort logging.")

        ### CSV SAVING ###
        self.control_started = False
        self.store_data = store_data
        self.csv_file = csv_filename
        self.row_limit = row_limit
        self.row_count = 0

        if self.store_data:
            self.init_csv()
            rospy.loginfo(f"Saving joint values into {self.csv_file}")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def init_csv(self):
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Command + state columns (unchanged)
                cmd_cols   = [f"cmd_{name}"   for name in self.joints_names]
                state_cols = [f"state_{name}" for name in self.joints_names]

                # Diagnostics columns: three metrics per motor, grouped by motor.
                # Layout: current_WRJ1, effort_WRJ1, cmd_effort_WRJ1, current_WRJ2, …
                diag_cols = []
                for name in self.diagnostics_joints:
                    diag_cols += [
                        f"diag_current_{name}",
                        f"diag_effort_{name}",
                        f"diag_cmd_effort_{name}",
                    ]

                writer.writerow(["Timestamp"] + cmd_cols + state_cols + diag_cols)

    def write_csv(self, timestamp, command_value, state_value, diag_values):
        """Write one row: timestamp | commands | states | diagnostics.

        diag_values is a list of dicts (one per joint, same order as joints_names).
        Fields that are unavailable (motorless joints or not yet received) are written
        as empty strings so the column count stays constant.
        """
        diag_flat = []
        for d in diag_values:
            diag_flat += [d["current"], d["effort"], d["cmd_effort"]]

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + list(command_value) + list(state_value) + diag_flat)

    # ------------------------------------------------------------------
    # Diagnostics callback
    # ------------------------------------------------------------------

    def diagnostics_callback(self, msg):
        """Parse DiagnosticArray and update the per-motor diagnostics buffer.

        Only status entries whose name starts with "rh SRDMotor " are processed.
        Motors with an empty values list (no physical motor) are skipped silently.
        """
        # Build a name -> values dict for fast lookup
        motor_data = {}
        for status in msg.status:
            if not status.name.startswith("rh SRDMotor "):
                continue
            if not status.values:          # motorless joint (e.g. FFJ1, FFJ2)
                continue
            kv = {kv_pair.key: kv_pair.value for kv_pair in status.values}
            motor_data[status.name] = kv

        # Map into the ordered buffer
        for i, motor_name in enumerate(self.diag_motor_names):
            kv = motor_data.get(motor_name)
            if kv is None:
                # Motor absent from this message: keep previous value rather than clearing,
                # so that CSV rows always carry the last known reading.
                continue
            self.latest_diag[i] = {
                "current":    kv.get("Measured Current",      ""),
                "effort":     kv.get("Measured Effort",       ""),
                "cmd_effort": kv.get("Last Commanded Effort", ""),
            }

        self.diag_received = True

    # ------------------------------------------------------------------
    # State callback (joint positions + CSV write)
    # ------------------------------------------------------------------

    def state_callback(self, msg):
        """Reorder joint positions and publish; optionally log to CSV."""
        new_joint_state = JointState()
        new_joint_state.header = msg.header

        joint_index_map = {name: i for i, name in enumerate(msg.name)}

        for joint_name in self.joint_names_ordered:
            if joint_name in joint_index_map:
                index = joint_index_map[joint_name]
                new_joint_state.name.append(joint_name)
                new_joint_state.position.append(msg.position[index])
                if len(msg.velocity) > 0:
                    new_joint_state.velocity.append(msg.velocity[index])
            else:
                rospy.logwarn(f"Joint '{joint_name}' not found in the received JointState message.")

        self.state_pub.publish(new_joint_state)

        if self.store_data and self.control_started and self.row_count < self.row_limit:
            # Snapshot the current diagnostics buffer (a shallow copy is enough because
            # each element is a plain dict replaced atomically by diagnostics_callback).
            diag_snapshot = list(self.latest_diag)
            self.write_csv(
                rospy.get_time(),
                self.joint_states,
                new_joint_state.position,
                diag_snapshot,
            )
            self.row_count += 1

    # ------------------------------------------------------------------
    # Command callback
    # ------------------------------------------------------------------

    def command_callback(self, msg):
        """Split Float64MultiArray command into per-joint topics with clamping and coupling."""
        data = msg.data

        if len(data) <= len(self.joint_command_publishers):
            rospy.logwarn(f"Received {len(data)} values, but expected at least {len(self.joint_command_publishers)}. Ignoring message.")
            return

        j = 0
        for i in range(len(self.joint_command_publishers)):
            if i in self.coupled_fingers:
                j1_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j1_value
                j += 1
                j2_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j2_value
                j += 1
                limited_value = j1_value + j2_value
                self.joint_command_publishers[i].publish(Float64(limited_value))
            else:
                limited_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = limited_value
                j += 1
                self.joint_command_publishers[i].publish(Float64(limited_value))

        if not self.control_started:
            self.control_started = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clamp(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def print_joint_states(self):
        while not rospy.is_shutdown():
            rospy.loginfo("\nJoint States:")
            for name, value in zip(self.joints_names, self.joint_states):
                rospy.loginfo(f"{name}: {value:.4f}")
            self.rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_data", type=str2bool, help="Boolean flag to enable or disable data storage")
    parser.add_argument("--csv_filename", type=str, default="shadowhand_joint_data.csv", help="Optional CSV filename")
    parser.add_argument("--row_limit", type=int, default=1000, help="Optional limit for the number of rows")

    args = parser.parse_args()

    try:
        node = ShadowHandBridge(args.store_data, args.csv_filename, args.row_limit)
        node.print_joint_states()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass