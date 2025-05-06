import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


# Constants to define training and visualisation.
GUI_MODE = False          # Set "True" to display pybullet in a window
EPISODE_LENGTH = 250      # Number of steps for one training episode
MAXIMUM_LENGTH = 1.8e6    # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6       # Increase of penalty by step_counter/PENALTY_STEPS

FAC_VELOCITY = 500       # Reward matching target velocity
FAC_DIRECTION = 500      # Reward matching target direction
FAC_STABILITY = 0.1       # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0      # Punish z movement of body
FAC_SLIP = 0.0            # Punish slipping of paws
FAC_ARM_CONTACT = 0.01    # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0        # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0        # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0       # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005      # Target height (m) of paw during swing phase

BOUND_ANG = 110         # Joint maximum angle (deg)
STEP_ANGLE = 11           # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1          # Improve angular velocity resolution before clip.

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0           # Percent
RANDOM_JOINT_ANGS = 0      # Percent
RANDOM_MASS = 0           # Percent, currently inactive
RANDOM_FRICTION = 0       # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
LENGTH_JOINT_HISTORY = 30 # Number of steps to store joint angles.

# Size of oberservation space is set up of: 
# [LENGTH_JOINT_HISTORY, quaternion, v_r, v_t, target_velocity]
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 4 + 3 + 3 + 3     


class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0  # 当前 episode 内的步数，到最大步数或者摔倒时归零
        self.step_counter_session = 0  # 累计整个训练过程中的总步数，episode 结束的时候会加入到总步数中
        self.state_history = np.array([])
        self.angle_history = np.array([])
        self.bound_ang = np.deg2rad(BOUND_ANG)  # 关节最大角度

        # # random target velocity per env
        # vx = np.random.uniform(-0.1, 0.1)
        # vy = np.random.uniform(-0.1, 0.1)
        # rz = np.random.uniform(-0.1, 0.1)
        # self.set_target_velocity(forward_velocity=vx,
        #                          lateral_velocity=vy,
        #                          angular_velocity=rz)
        # fixed for debugging
        self.set_target_velocity(forward_velocity=0.0,
                                 lateral_velocity=0.0,
                                 angular_velocity=1.0)

        if GUI_MODE:
            p.connect(p.GUI)
            # Uncommend to create a video.
            #video_options = ("--width=960 --height=540 
            #                + "--mp4=\"training.mp4\" --mp4fps=60")
            #p.connect(p.GUI, options=video_options) 
        else:
            # Use for training without visualisation (significantly faster).
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, 
                                     cameraYaw=-170, 
                                     cameraPitch=-40, 
                                     cameraTargetPosition=[0.4,0,0])

        # The action space are the 8 joint angles. Box 是一种用于表示连续动作空间的类型（每个维度有上下界）
        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))

        # The observation space are the torso roll, pitch and the 
        # angular velocities and a history of the last 30 joint angles.
        self.observation_space = gym.spaces.Box(np.array([-1]*SIZE_OBSERVATION), 
                                                np.array([1]*SIZE_OBSERVATION))


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0:3]  # 位置 (3)
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id),
                                                   dtype=object)[:,0]  # 关节角度 (8)
        ds = np.deg2rad(STEP_ANGLE) # Maximum change of angle per step
        joint_angs += action * ds # Change per step including agent action

        # Apply joint boundaries individually.
        min_ang = -self.bound_ang
        max_ang = self.bound_ang
        joint_angs[0] = np.clip(joint_angs[0], min_ang, max_ang) # shoulder_left
        joint_angs[1] = np.clip(joint_angs[1], min_ang, max_ang) # elbow_left
        joint_angs[2] = np.clip(joint_angs[2], min_ang, max_ang) # shoulder_right
        joint_angs[3] = np.clip(joint_angs[3], min_ang, max_ang) # elbow_right
        joint_angs[4] = np.clip(joint_angs[4], min_ang, max_ang) # hip_right
        joint_angs[5] = np.clip(joint_angs[5], min_ang, max_ang) # knee_right
        joint_angs[6] = np.clip(joint_angs[6], min_ang, max_ang) # hip_left
        joint_angs[7] = np.clip(joint_angs[7], min_ang, max_ang) # knee_left

        # Transform angle to degree and perform rounding, because 
        # OpenCat robot have only integer values.
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        joint_angs = np.deg2rad(joint_angsDegRounded)

        # Simulate delay for data transfer. Delay has to be modeled to close 
        # "reality gap").
        p.stepSimulation()

        # Check for friction of paws, to prevent slipping while training. 爪子如果接触，就尽量不要滑动
        paw_contact = []
        paw_idx = [3, 6, 9, 12]  # 四个爪子在 URDF 中的 linkIndex
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robot_id, 
                                                          linkIndexA=idx)  # TODO 可能要加上 bodyB=plane_id
                                    else False)

        paw_slipping = 0
        for in_contact in np.nonzero(paw_contact)[0]:
            paw_slipping += np.linalg.norm((
                            p.getLinkState(self.robot_id,
                                           linkIndex=paw_idx[in_contact], 
                                           computeLinkVelocity=1)[0][0:1]))  # 算对应的滑动速度

        # Read clearance of paw from ground
        paw_clearance = 0
        for idx in paw_idx:
            paw_z_pos = p.getLinkState(self.robot_id, linkIndex=idx)[0][2]
            paw_clearance += (paw_z_pos-PAW_Z_TARGET)**2 * np.linalg.norm(
                (p.getLinkState(self.robot_id, linkIndex=idx, 
                                computeLinkVelocity=1)[0][0:1]))**0.5  # 在摆动阶段要求更高的抬腿高度

        # Check if elbows or lower arm are in contact with ground 其他部分不要接触地面
        arm_idx = [1, 2, 4, 5]
        for idx in arm_idx:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx):
                self.arm_contact += 1

        # Read clearance of torso from ground
        base_clearance = p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # Set new joint angles
        p.setJointMotorControlArray(self.robot_id, 
                                    self.joint_id, 
                                    p.POSITION_CONTROL, 
                                    joint_angs, 
                                    forces=np.ones(8)*0.2)
        p.stepSimulation() # Delay of data transfer

        # Normalize joint_angs
        joint_angs[0] /= self.bound_ang
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Adding every 2nd angle to the joint angle history.
        if(self.step_counter % 2 == 0):
            self.angle_history = np.append(self.angle_history, 
                                           self.randomize(joint_angs, 
                                                          RANDOM_JOINT_ANGS))
            self.angle_history = np.delete(self.angle_history, np.s_[0:8])  # 删掉最前面的 8 个

        self.recent_angles = np.append(self.recent_angles, joint_angs)  # 这个每次更新
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]  # 总长是 3，所以把前面两个拿出来
        joint_angs_prev_prev = self.recent_angles[0:8]

        # Read robot state (pitch, roll and their derivatives of the torso)
        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        p.stepSimulation() # Emulated delay of data transfer via serial port
        roll, pitch, yaw = p.getEulerFromQuaternion(state_ang)
        
        state_vel_r = np.asarray(p.getBaseVelocity(self.robot_id)[1])  # 返回 v 和 w，取后者
        state_vel_r = state_vel_r*ANG_FACTOR  # 一个固定比例
        state_vel_r_clip = np.clip(state_vel_r, -1, 1)
        state_vel_t = np.asarray(p.getBaseVelocity(self.robot_id)[0])  # 返回 v 和 w，取前者
        state_vel_t_clip = np.clip(state_vel_t, -1, 1)

        # 更新机器人状态
        self.state_robot = np.concatenate((state_ang,
                                           state_vel_r_clip,
                                           state_vel_t_clip,
                                           self.target_velocity))

        # Penalty and reward
        smooth_movement = np.sum(
            FAC_SMOOTH_1*np.abs(joint_angs-joint_angs_prev)**2
            + FAC_SMOOTH_2*np.abs(joint_angs
            - 2*joint_angs_prev 
            + joint_angs_prev_prev)**2)

        z_velocity = p.getBaseVelocity(self.robot_id)[0][2]

        body_stability = (FAC_STABILITY * (state_vel_r_clip[0]**2 
                                          + state_vel_r_clip[1]**2) 
                                          + FAC_Z_VELOCITY * z_velocity**2)
        
        # 当前速度向量
        vx, vy = state_vel_t_clip[0], state_vel_t_clip[1]  # 世界坐标系下的速度
        vx_robot =  np.cos(yaw) * vx + np.sin(yaw) * vy
        vy_robot = -np.sin(yaw) * vx + np.cos(yaw) * vy
        
        # 计算速度误差（向量差的范数）
        velocity_error = np.linalg.norm(vx_robot - self.target_velocity[0]) + np.linalg.norm(vy_robot - self.target_velocity[1])
        angular_error = np.linalg.norm(state_vel_r_clip[2] - self.target_velocity[2])
        
        # 速度匹配奖励
        velocity_reward = -FAC_VELOCITY * velocity_error
        angular_reward = -FAC_DIRECTION * angular_error
        
        # 总奖励
        reward = (velocity_reward
                 + angular_reward
                 - self.step_counter_session/PENALTY_STEPS * (
                    smooth_movement + body_stability 
                    + FAC_CLEARANCE * paw_clearance 
                    + FAC_SLIP * paw_slipping**2 
                    + FAC_ARM_CONTACT * self.arm_contact))

        # Set state of the current state.
        terminated = False
        truncated = False
        info = {}

        # Stop criteria of current learning episode: 
        # Number of steps or robot fell.
        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            terminated = False
            truncated = True

        elif self.is_fallen(): # Robot fell
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True
            truncated = False

        self.observation = np.hstack((self.state_robot, self.angle_history))

        return (np.array(self.observation).astype(np.float32), 
                        reward, terminated, truncated, info)


    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.arm_contact = 0  # 手肘和地面的接触（用于惩罚）
        p.resetSimulation()
        # Disable rendering during loading.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        plane_id = p.loadURDF("plane.urdf")
        # # 加载自定义地面mesh
        # ground_shape = p.createCollisionShape(p.GEOM_MESH, 
        #                                     fileName="models/Martian-Terrain.stl",
        #                                     meshScale=[0.03,0.03,0.03])
        # ground_visual = p.createVisualShape(p.GEOM_MESH,
        #                                   fileName="models/Martian-Terrain.stl",
        #                                   meshScale=[0.03,0.03,0.03])
        # ground_id = p.createMultiBody(baseMass=0,
        #                             baseCollisionShapeIndex=ground_shape,
        #                             baseVisualShapeIndex=ground_visual,
        #                             basePosition=[-1.0,1.0,-1.2],
        #                             baseOrientation=p.getQuaternionFromEuler([90,0,0]))

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])  # 是一个四元数（scalar-last）

        urdf_path = "models/"#"/content/drive/My Drive/opencat-gym-esp32/models/"
        self.robot_id = p.loadURDF(urdf_path + "bittle_esp32.urdf", 
                                   start_pos, start_orient, 
                                   flags=p.URDF_USE_SELF_COLLISION) 
        
        # Initialize urdf links and joints.  从 URDF 中获取 8 个可控的关节
        self.joint_id = []
        #paramIds = []
        for j in range(p.getNumJoints(self.robot_id)):  # 13 个关节（但有些是 fixed 的）
            info = p.getJointInfo(self.robot_id, j)
            joint_name = info[1]
            joint_type = info[2]

            if (joint_type == p.JOINT_PRISMATIC   # 1
                or joint_type == p.JOINT_REVOLUTE):  # 0
                self.joint_id.append(j)
                #paramIds.append(p.addUserDebugParameter(joint_name.decode("utf-8")))
                # Limiting motor dynamics. Although bittle's dynamics seem to 
                # be be quite high like up to 7 rad/s.
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10)  # 在仿真中限制关节最大速度
        
        # Setting start position. This influences training.
        joint_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*50)  # 分别给 0° 和 50°

        i = 0
        for j in self.joint_id:
            p.resetJointState(self.robot_id,j, joint_angs[i])
            i = i+1

        # Normalize joint angles.
        joint_angs[0] /= self.bound_ang  # 最大关节角度 110 度，这样放在 -1 和 1 之间
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Read robot state (pitch, roll and their derivatives of the torso)
        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]  # 返回 pos + orient，取后者
        state_vel_r = np.asarray(p.getBaseVelocity(self.robot_id)[1])  # 返回 v 和 w，取后者
        state_vel_r = state_vel_r*ANG_FACTOR  # 一个固定比例
        state_vel_t = np.asarray(p.getBaseVelocity(self.robot_id)[0])  # 返回 v 和 w，取前者
        
        self.state_robot = np.concatenate((state_ang,  # （4）
                                           np.clip(state_vel_r, -1, 1),  # （3）
                                           np.clip(state_vel_t, -1, 1),  # （3）
                                           self.target_velocity))  # （3）

        # Initialize robot state history with reset position
        state_joints = np.asarray(  # getJointStates 会拿到关节的位置、速度、反作用力、力矩；最后只保留位置
            p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]  # ! 这里 object 的 type 会把每个元素当做对象来处理，不进行强制转换
        state_joints /= self.bound_ang  # 最大关节角度 110 度，这样放在 -1 和 1 之间
        
        self.angle_history = np.tile(state_joints, LENGTH_JOINT_HISTORY)  # 重复 30 次，用来作为观测空间
        self.recent_angles = np.tile(state_joints, LENGTH_RECENT_ANGLES)  # 重复 3 次，用来计算平滑
        self.observation = np.concatenate((self.state_robot,   # （13）
                                           self.angle_history))  # （30*8）
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return np.array(self.observation).astype(np.float32), info


    def render(self, mode='human'):
        pass


    def close(self):
        p.disconnect()


    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", 
            when pitch or roll is more than 1.3 rad.
        """
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        is_fallen = (np.fabs(orient[0]) > 1.3 
                    or np.fabs(orient[1]) > 1.3)

        return is_fallen


    def randomize(self, value, percentage):
        """ Randomize value within percentage boundaries.
        """
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))

        return value_randomized
        
    def set_target_velocity(self, forward_velocity, lateral_velocity=0.0, angular_velocity=0.0):
        """设置目标速度和角速度
        
        Args:
            forward_velocity (float): 目标前进速度 (x方向)
            lateral_velocity (float): 目标横向速度 (y方向)
            angular_velocity (float): 目标角速度 (绕z轴)
        """
        self.target_velocity = np.array([forward_velocity, lateral_velocity, angular_velocity])
