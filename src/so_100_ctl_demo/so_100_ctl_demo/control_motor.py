import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time


class TestROS2Bridge(Node):
    def __init__(self):

        super().__init__("test_ros2bridge")

        # 创建发布者。此发布者将发布 JointState 消息到 /joint_command 主题。
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)

        # 创建 JointState 消息
        self.joint_state = JointState()

        self.joint_state.name = [
            "Rotation",
            "Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
            "Jaw"
        ]

        num_joints = len(self.joint_state.name)

        # 确保 Kit 的编辑器处于播放状态以接收消息
        self.joint_state.position = np.array([0.0] * num_joints, dtype=np.float64).tolist()
        self.default_joints = [0, 0, 0, 0, 0, 0]

        # 将运动限制在较小范围内（这不是机器人的范围，仅是运动范围）
        self.max_joints = np.array(self.default_joints) + 0.3
        self.min_joints = np.array(self.default_joints) - 0.3

        # 使用位置控制让机器人围绕每个关节摆动
        self.time_start = time.time()

        timer_period = 0.05  # 秒
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_position = (
            np.sin(time.time() - self.time_start) * (self.max_joints - self.min_joints) * 0.5 + self.default_joints
        )
        self.joint_state.position = joint_position.tolist()

        # 将消息发布到主题
        self.publisher_.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)
    ros2_publisher = TestROS2Bridge()
    rclpy.spin(ros2_publisher)
    ros2_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
