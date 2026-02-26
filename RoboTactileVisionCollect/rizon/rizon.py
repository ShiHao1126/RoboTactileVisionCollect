import time
import spdlog
import flexivrdk
import numpy as np
from typing import Optional, List, Callable
from scipy.spatial.transform import Rotation as R
from RoboTactileVisionCollect.utils import mat2xyzquat, delta_xyz, delta_quat, quat2eulerZYX, Colors
logger = spdlog.ConsoleLogger("Rizon")


RIZON_ID = 'Rizon4s-063231'

RIZON_MAX_CONTACT_FORCE = 30             # N, maximum allowed contact force (increased from 20 to 30)
RIZON_MAX_CONTACT_TORQUE = 5             # Nm, maximum allowed contact torque

RIZON_TCP_MAX_VEL = (0.08, 0.2)        # m/s, maximum TCP linear/angular velocity
RIZON_TCP_MAX_ACC = (0.1, 0.2)       # m/s^2, maximum TCP linear/angular acceleration
RIZON_TCP_POSE_EPSILON = (0.001, 0.01) # Position tolerance 1mm, orientation tolerance 0.001rad

RIZON_JOINT_MAX_VEL = 0.1            # rad/s, typical safe range 0.2~0.7
RIZON_JOINT_MAX_ACC = 0.1            # rad/s², typical safe range 0.5~2.0
RIZON_JOINT_EPSILON = 0.01           # rad, ~1°, moderate tolerance
RIZON_HOME_JOINTS = np.deg2rad([0, -40, 0, 90, 0, 40, 0])


class Rizon:
    def __init__(self, tool_name: str = "Flange"):
        logger.info(f"Initializing Rizon robot with tool: {tool_name}")
        self.tool_name = tool_name
        self.DOF = 7

        self.robot = flexivrdk.Robot(RIZON_ID)
        info = self.robot.info()
        self._joint_limits = (np.array(info.q_min), np.array(info.q_max))

        # clear fault if any
        if self.robot.fault():
            logger.warn("Fault occurred on the connected robot, trying to clear ...")
            if not self.robot.ClearFault():
                logger.error("Fault cannot be cleared, exiting ...")
                return 1
            logger.info("Fault on the connected robot is cleared")

        # enable robot
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(1)
        logger.info("Robot is now operational")
            
        # set tool
        self.robot.SwitchMode(flexivrdk.Mode.IDLE)
        tool = flexivrdk.Tool(self.robot)
        if tool.exist(tool_name):
            tool.Switch(tool_name)
        else:
            raise RuntimeError(f"{tool_name} tool not found")

        # zero ft sensor
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.1)
        logger.debug("FT sensor zeroed")

    def close(self) -> None:
        """释放机器人连接资源"""
        try:
            if hasattr(self, "robot") and self.robot is not None:
                try:
                    # 尝试切换到IDLE模式，停止所有运动
                    try:
                        self.robot.SwitchMode(flexivrdk.Mode.IDLE)
                    except Exception:
                        pass
                    
                    # 停止机器人
                    if self.robot.connected():
                        try:
                            self.robot.Stop()
                        except Exception:
                            pass
                        
                        # 清除故障
                        try:
                            if self.robot.fault():
                                self.robot.ClearFault()
                        except Exception:
                            pass
                except Exception as e:
                    logger.warn(f"Error during robot cleanup: {e}")
        except Exception as e:
            logger.warn(f"Error checking robot state: {e}")
        finally:
            # 确保引用被清除
            self.robot = None
            logger.debug("Robot connection closed")

    def hold(
        self,
        duration_before: float = 1.0,
        duration_after: float = 1.0,
        callback: Optional[Callable[["Rizon"], None]] = None,
    ) -> None:
        total_duration = duration_before + duration_after
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("Hold", dict())
        start = time.time()
        callback_called = False

        while True:
            elapsed = time.time() - start
            if elapsed >= total_duration:
                break

            state = self.robot.primitive_states()
            if state.get("terminated", 0):
                self.robot.ExecutePrimitive("Hold", dict())

            if (not callback_called) and (elapsed >= duration_before) and callback is not None:
                try:
                    self.robot.ExecutePrimitive("Hold", dict())
                    callback(self)
                except Exception as e:
                    logger.error(f"Callback in hold() failed: {e}")
                callback_called = True

            if self.robot.fault():
                raise RuntimeError("Fault during Hold")
            time.sleep(0.01)

        logger.info(
            Colors.colorize("✓ Hold END", Colors.GREEN, bold=True)
            + f" | Time: {elapsed:.2f}s"
        )
        
    def move_l(
        self, 
        target_pose: np.ndarray,    # [x, y, z, q_w, q_x, q_y, q_z] (list/np.ndarray)
        vel: float = 0.001, 
        acc: float = 0.1, 
        angVel: float = 150.0,
        zoneRadius: str = "Z1", 
        targetTolerLevel: int = 1
        ) -> None:
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)


        pose7 = np.asarray(target_pose, dtype=float).reshape(-1)
        if pose7.size != 7:
            raise ValueError(
                f"move_l expects 7D [x,y,z,qw,qx,qy,qz], got shape {np.asarray(target_pose).shape}"
            )
        xyz = pose7[:3]  # [x, y, z] in m
        quat_wxyz = pose7[3:7]  # [q_w, q_x, q_y, q_z]

        # 四元数符号连续化，避免 q 和 -q 等价导致欧拉角跳变 -> 快速旋转
        try:
            curr_quat = np.asarray(self.robot.states().tcp_pose[3:7], dtype=float)
            if np.dot(quat_wxyz, curr_quat) < 0:
                quat_wxyz = -quat_wxyz
        except Exception:
            if quat_wxyz[0] < 0:
                quat_wxyz = -quat_wxyz

        euler_deg = quat2eulerZYX(quat_wxyz.tolist(), degree=True)  # [rot_x, rot_y, rot_z] in deg
        coord = flexivrdk.Coord(
            xyz.tolist(),          # m
            euler_deg,             # deg
            ["WORLD", "WORLD_ORIGIN"],
        )

        start_time = time.time()
        self.robot.ExecutePrimitive(
            "MoveL",
            {
                "target": coord,
                "acc": acc,
                "vel": vel,
                "angVel": angVel,
                "zoneRadius": zoneRadius,
                "targetTolerLevel": targetTolerLevel,
            },
        )

        while True:
            st = self.robot.primitive_states()
            if st.get("reachedTarget", 0):
                break
            if self.robot.fault():
                raise RuntimeError("Fault during MoveL")
            time.sleep(0.05)

        # for logging
        pos_target_mm = (np.array(xyz) * 1000).tolist()
        pos_error_mm = ((np.array(self.robot.states().tcp_pose[:3]) - np.array(xyz)) * 1000).tolist()
        
        logger.info(Colors.colorize(f"✓ MoveL END", Colors.GREEN, bold=True) +
         f"Time: {(time.time() - start_time):.2f}s | "
         f"Pos target: [{pos_target_mm[0]:.2f}, {pos_target_mm[1]:.2f}, {pos_target_mm[2]:.2f}]mm | "
         f"Pos error: [{pos_error_mm[0]:.2f}, {pos_error_mm[1]:.2f}, {pos_error_mm[2]:.2f}]mm")

    def contact(
        self,
        contactCoord: str = "world",
        contactDir: List[float] = [0.0, 0.0, -1.0],
        contactVel: float = 0.001,
        maxContactForce: float = 1.0,
        enableFineContact: int = 0
    ) -> None:
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # Zero ft sensor
        logger.debug("Zeroing force/torque sensors...")
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.1)
        logger.debug("Sensor zeroing complete")

        start_time = time.time()
        self.robot.ExecutePrimitive("Contact", {
            "contactCoord": contactCoord,
            "contactDir": contactDir,
            "contactVel": contactVel,
            "maxContactForce": maxContactForce,
            "enableFineContact": enableFineContact,
        })
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.1)

        # for logging
        final_force = np.array(self.robot.states().ext_wrench_in_world[:3]).tolist()
        
        logger.info(Colors.colorize(f"✓ Contact END", Colors.GREEN, bold=True) +
         f"Time: {(time.time() - start_time):.2f}s | "
         f"Contact force: {maxContactForce:.2f}N | "
         f"Final force: [{final_force[0]:.2f}, {final_force[1]:.2f}, {final_force[2]:.2f}]N")

    def force_hybrid(
        self,
        target_pose: np.ndarray,
        target_wrench: np.ndarray,
        force_axis: List[int] = [0, 0, 1, 0, 0, 0],
        force_coord: str = "world",
        enable_max_wrench: List[int] = [0, 0, 1, 0, 0, 0],
        max_contact_wrench: List[float] = [25.0, 25.0, 25.0, 5.0, 5.0, 5.0],
        max_vel_force_dir: List[float] = [0.005, 0.005, 0.005],
        vel: float = 0.001,
        acc: float = 0.1,
        targetTolerLevel: int = 1
    ) -> None:
        assert target_pose is not None, "target_pose must be provided"
        assert target_wrench is not None, "target_wrench must be provided"
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # zero ft sensor
        logger.debug("Zeroing force/torque sensors...")
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.1)
        logger.debug("Sensor zeroing complete")

        # 转换 target_pose 为 COORD 格式（仿照 move_l）
        pose7 = np.asarray(target_pose, dtype=float).reshape(-1)
        if pose7.size != 7:
            raise ValueError(
                f"force_hybrid expects 7D [x,y,z,qw,qx,qy,qz], got shape {np.asarray(target_pose).shape}"
            )
        xyz = pose7[:3]  # [x, y, z] in m
        quat_wxyz = pose7[3:7]  # [q_w, q_x, q_y, q_z]

        # 四元数符号连续化，避免 q 和 -q 等价导致欧拉角跳变 -> 快速旋转
        try:
            curr_quat = np.asarray(self.robot.states().tcp_pose[3:7], dtype=float)
            if np.dot(quat_wxyz, curr_quat) < 0:
                quat_wxyz = -quat_wxyz
        except Exception:
            if quat_wxyz[0] < 0:
                quat_wxyz = -quat_wxyz

        euler_deg = quat2eulerZYX(quat_wxyz.tolist(), degree=True)  # [rot_x, rot_y, rot_z] in deg
        coord = flexivrdk.Coord(
            xyz.tolist(),          # m
            euler_deg,             # deg
            ["WORLD", "WORLD_ORIGIN"],
        )

        # 转换 force_coord 为 COORD 格式（如果 force_coord 是字符串 "world"，则使用 world 坐标系）
        if force_coord == "world":
            force_coord_coord = flexivrdk.Coord(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                ["WORLD", "WORLD_ORIGIN"],
            )
        elif force_coord == "tcp":
            # TCP 坐标系：使用当前 TCP 位姿
            current_pose = self.robot.states().tcp_pose
            current_xyz = current_pose[:3]
            current_quat = current_pose[3:7]
            current_euler = quat2eulerZYX(current_quat.tolist(), degree=True)
            force_coord_coord = flexivrdk.Coord(
                current_xyz.tolist(),
                current_euler,
                ["TCP", "ONLINE"],
            )
        else:
            # 默认使用 world 坐标系
            force_coord_coord = flexivrdk.Coord(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                ["WORLD", "WORLD_ORIGIN"],
            )

        # force hybrid
        start_time = time.time()
        self.robot.ExecutePrimitive("ForceHybrid", {
            "target": coord,
            "targetWrench": target_wrench.tolist() if isinstance(target_wrench, np.ndarray) else target_wrench,
            "forceAxis": force_axis,
            "forceCoord": force_coord_coord,
            "enableMaxWrench": enable_max_wrench,
            "maxContactWrench": max_contact_wrench,
            "maxVelForceDir": max_vel_force_dir,
            "vel": vel,
            "acc": acc,
            "targetTolerLevel": targetTolerLevel,
        })

        force_tolerance = 0.05  # 力控精度（N）
        max_wait_time = 15.0  # 最大等待时间（s）
        force_success_count = 0
        force_success_threshold = 10

        while True:                        
            current_force = np.array(self.robot.states().ext_wrench_in_world[:3])
            for i in range(3):
                if force_axis[i] == 1:  # 如果是力控轴
                    target_force = target_wrench[i]
                    if abs(current_force[i] - target_force) <= force_tolerance:
                        force_success_count += 1
                        break
            if force_success_count >= force_success_threshold:
                break

            if time.time() - start_time > max_wait_time:
                logger.warn(f"ForceHybrid timeout after {max_wait_time}s")
            if self.robot.fault():
                raise RuntimeError("Fault during ForceHybrid")
            time.sleep(0.05)

        # for logging
        final_force = np.array(self.robot.states().ext_wrench_in_world[:3]).tolist()
        logger.info(Colors.colorize(f"✓ ForceHybrid END", Colors.GREEN, bold=True) +
         f"Time: {(time.time() - start_time):.2f}s | "
         f"Final force: [{final_force[0]:.2f}, {final_force[1]:.2f}, {final_force[2]:.2f}]N")

    def get_rot_pose(
        self,
        angle: float,
        axis: str = "z",
    ) -> np.ndarray:
        pose = np.array(self.robot.states().tcp_pose, dtype=float).copy()
        pos = pose[:3]
        quat_wxyz = pose[3:7]  # [w, x, y, z]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float) # [w, x, y, z] -> [x, y, z, w] (scipy)
        rot = R.from_quat(quat_xyzw)
        
        axis_map = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }
        if axis.lower() not in axis_map:
            raise ValueError(f"axis must be 'x','y','z', but got '{axis}'")
        rot_vec = axis_map[axis.lower()] * np.deg2rad(angle)
        rot_delta = R.from_rotvec(rot_vec)

        # apply rotation
        new_rot = rot * rot_delta
        new_quat_xyzw = new_rot.as_quat()
        new_quat_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]], dtype=float) # [x, y, z, w] -> [w, x, y, z]

        # avoid the quaternion sign change
        if new_quat_wxyz[0] < 0:
            new_quat_wxyz = -new_quat_wxyz

        return np.concatenate([pos, new_quat_wxyz])
        
    def force_comp(
        self,
        target_pose: np.ndarray,    # [x, y, z, q_w, q_x, q_y, q_z] (list/np.ndarray)
        stiff_scale: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vel: float = 0.01,
        acc: float = 0.1,
        angVel: float = 15.0,
        targetTolerLevel: int = 5,
        enable_max_wrench: List[int] = [1, 1, 1, 1, 1, 1],
        max_contact_wrench: List[float] = [25.0, 25.0, 25.0, 5.0, 5.0, 5.0],
    ) -> None:
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # convert target_pose to coord
        pose7 = np.asarray(target_pose, dtype=float).reshape(-1)
        if pose7.size != 7:
            raise ValueError(
                f"move_l expects 7D [x,y,z,qw,qx,qy,qz], got shape {np.asarray(target_pose).shape}"
            )
        xyz = pose7[:3]  # [x, y, z] in m
        quat_wxyz = pose7[3:7]  # [q_w, q_x, q_y, q_z]
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz
        euler_deg = quat2eulerZYX(quat_wxyz.tolist(), degree=True)  # [rot_x, rot_y, rot_z] in deg
        coord = flexivrdk.Coord(
            xyz.tolist(),          # m
            euler_deg,             # deg
            ["WORLD", "WORLD_ORIGIN"],
        )

        self.robot.ExecutePrimitive("ForceComp", {
            "target": coord,
            "stiffScale": stiff_scale,
            "vel": vel,
            "acc": acc,
            "angVel": angVel,
            "targetTolerLevel": targetTolerLevel,
            "enableMaxWrench": enable_max_wrench,
            "maxContactWrench": max_contact_wrench,
        })
