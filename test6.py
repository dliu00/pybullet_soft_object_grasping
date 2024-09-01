import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
# 连接到PyBullet仿真环境
p.connect(p.GUI) #启动图形界面
# physicsClient = p.connect(p.DIRECT) #不启动图形界面
# 设置仿真环境的资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
camera_distance = 1.0  # 可视化界面与目标点的距离
camera_yaw = 80        # 可视化界面的水平旋转角度（以目标点为中心）
camera_pitch = -10     # 可视化界面的俯仰角度
camera_target_position = [0.3, 0.5-0.01,0]  # 可视化界面对准的目标点（通常是机器人的位置）

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

# 设置重力
p.setGravity(0, 0, -9.8)
# 加载平面和机械臂
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=5)
robot_id = p.loadURDF(r"ros_kortex\myrobo.urdf", useFixedBase=True)
#夹爪摩擦
for link_index in [6,7,8,9,10,11]:
    p.changeDynamics(robot_id, link_index, lateralFriction=1)
p.changeDynamics(robot_id, 7, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1)
p.changeDynamics(robot_id, 9, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1)
p.changeDynamics(robot_id, 11, lateralFriction=1, spinningFriction=0.1, rollingFriction=0.1)

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output
    
class ExponentialMovingAverage:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.previous_value = None

    def filter(self, value):
        if self.previous_value is None:
            filtered_value = value
        else:
            filtered_value = self.alpha * value + (1 - self.alpha) * self.previous_value
        self.previous_value = filtered_value
        return filtered_value
    
def get_joint_info(robot_id):
    num_joints = p.getNumJoints(robot_id)

    # 遍历每个关节并打印信息
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        print(f"Joint {joint_index}:")
        print(f"  Name: {joint_info[1].decode('utf-8')}")
        print(f"  Type: {joint_info[2]}")
        print(f"  Lower limit: {joint_info[8]}")
        print(f"  Upper limit: {joint_info[9]}")
        print(f"  Max force: {joint_info[10]}")
        print(f"  Max velocity: {joint_info[11]}")
        print()

def calculate_combined_friction_indicator(robot_id, link_indices, object_friction, gripper_friction):
    """
    计算多个接触点的合成摩擦锥指标，检查总力是否在摩擦锥内。
    
    参数:
    robot_id (int): 机器人URDF的ID。
    link_indices (list): 包含手指链接索引的列表。
    object_friction (float): 物体的摩擦系数。
    gripper_friction (float): 机械臂（夹爪）的摩擦系数。

    返回:
    dict: 包含整体摩擦锥检查的结果和总力信息的字典。
    """
    mu = min(object_friction, gripper_friction)  # 取物体和夹爪摩擦系数的较小值
    total_normal_force = 0
    total_tangential_force = np.array([0.0, 0.0, 0.0])  # 初始化总切向力向量
    
    for link_index in link_indices:
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)
        
        for contact in contact_points:
            normal_force = contact[9]  # 法向力大小
            lateral_friction1 = contact[10]  # 切向摩擦力1大小
            lateral_friction_dir1 = np.array(contact[11])  # 切向摩擦力1方向向量
            lateral_friction2 = contact[12]  # 切向摩擦力2大小
            lateral_friction_dir2 = np.array(contact[13])  # 切向摩擦力2方向向量
            
            # 合成切向摩擦力
            lateral_force_total = (lateral_friction1 * lateral_friction_dir1 +
                                   lateral_friction2 * lateral_friction_dir2)
            
            # 累加到总的法向力和总的切向摩擦力向量
            total_normal_force += normal_force
            total_tangential_force += lateral_force_total
    
    # 计算总切向力的大小
    total_tangential_magnitude = np.linalg.norm(total_tangential_force)
    
    # 检查总力是否在摩擦锥内
    is_within_friction_cone = total_tangential_magnitude <= mu * total_normal_force

    return is_within_friction_cone

def get_normal_forces(robot_id, link_indices=[7,9,11]):
    """
    获取指定链接的总体法向力

    参数:
        robot_id: 机器人或物体的唯一ID
        link_indices: 包含链接索引的列表（例如[7, 9, 11]）

    返回:
        一个字典，包含每个链接索引及其对应的总体法向力
    """
    # 初始化字典，用于存储每个链接的总体法向力
    average_forces = {7: 0, 9: 0, 11: 0}
    
    # 遍历要检查的链接
    for link_index in link_indices:
        # 获取特定链接上的接触点信息
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)
        
        # 累加法向力
        for point in contact_points:
            average_forces[link_index] += point[9]  # 累加法向力
            # average_forces[link_index] = average_forces[link_index]/len(contact_points)
    
    return average_forces

def calculate_average_contact_velocity(robot_id, finger_links):
    """
    计算每根手指接触点的平均速度
    
    参数:
        robot_id: 机器人或夹爪的唯一ID
        finger_links: 需要计算的手指链接索引列表
    
    返回:
        一个字典，包含每根手指的接触点平均速度
    """
    average_velocities = {}

    # 遍历每根手指的链接
    for link_index in finger_links:
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)

        # 如果没有接触点，跳过计算
        if len(contact_points) == 0:
            average_velocities[link_index] = np.array([0.0, 0.0, 0.0])
            continue

        # 初始化速度和计数器
        total_velocity = np.array([0.0, 0.0, 0.0])
        count = 0

        # 计算每个接触点的相对速度
        for point in contact_points:
            relative_velocity = np.array(point[11])  # 获取接触点的相对速度向量
            total_velocity += relative_velocity
            count += 1

        # 计算平均速度
        average_velocity = total_velocity / count if count > 0 else np.array([0.0, 0.0, 0.0])
        average_velocities[link_index] = average_velocity

    return average_velocities

def calculate_soft_body_center_of_mass(soft_body_id):
    """
    计算给定软体体的质心位置。

    参数:
    soft_body_id (int): 软体体的ID。

    返回:
    list: 质心的位置，格式为 [x, y, z]。
    """
    # 获取软体体的顶点（节点）数据
    mesh_data = p.getMeshData(soft_body_id, flags=p.MESH_DATA_SIMULATION_MESH)
    node_positions = mesh_data[1]
    
    # 确保节点数据不为空
    num_nodes = len(node_positions)
    if num_nodes == 0:
        raise ValueError("No nodes found in the soft body.")
    
    # 计算质心位置
    center_of_mass = [sum(pos[i] for pos in node_positions) / num_nodes for i in range(3)]
    
    return center_of_mass

def gripper_control(robot_id, grasp_position, put_position, only_lift, lift_height=0.4, pid_params=(2,0.001,0.03), max_torque=10, mode=p.TORQUE_CONTROL,ran=100,time_step=1/1000):
    closing_joints = [6, 8, 10]
    new_closing_joints = [7,9,11]
    gripper_joints_index_list = [6,7,8,9,10,11]
    print("grasper begin!!!!")
    pass_6_8_10 = 0
    soft_masscenter_now = [0,0,0]
    soft_masscenter_pre = [0,0,0]
    count_position = 0  #确定是否是保持的位置
    softbody_vel = 0
    cone_indicator = 0  #确定是否达到摩擦锥条件
    gripper_close = 0 #夹爪闭合判断
    gripper_6_8_10_refresh = 0 #从position控制到torque控制切换
    proportion_dic = {7:0.0,9:0.0,11:0.0}  #三个手指力的控制比例
    jointangle_dic = {7:0.0,9:0.0,11:0.0}

    force_joint = 100

    frame_counter = 0
    frames_between_adjustments = 2
    previous_average_z_velocity = 0.0
    applied_force = 0.0 # 记录上次应用的力
    previous_applied_force = 0.0
    pid = PIDController(*pid_params)
    dt = time_step
    force_record = np.array([])
    force_record_filter = np.array([])
    time_force = np.array([])
    distance_record = np.array([])

    final_x_up = grasp_position[0]
    final_y_up = grasp_position[1]
    # final_y_up = final_y_up - 0.001
    final_z_up = lift_height

    final_x_horizontal = put_position[0]
    final_y_horizontal = put_position[1]
    # final_y_horizontal = final_y_horizontal - 0.01
    final_z_horizontal = lift_height

    final_x_down = final_x_horizontal
    final_y_down = final_y_horizontal
    final_y_down = final_y_down - 0.01
    final_z_down = put_position[2]

    calcu_height_list = 0   #数值运动分割出的列表
    calcu_horizontal_list = 0
    calcu_down_list = 0

    count_height = 0  #竖直运动计数器
    count_horizontal = 0
    count_down = 0
    count_finish = 0
    count_up_start = 0

    num_height = 100  #竖直运动时路径分割的份数
    num_x_y = 100

    up_indicator = 0 #判断上升运动是否完成
    down_indicator = 0 #判断下降运动是否完成
    horizontal_indicator = 0 #判断平移运动是否完成
    all_finish_indicator = 0
    total_contact_force = 0 #物体所有接触点力的和

    ema_filter = ExponentialMovingAverage(alpha=0.1)

    ema_filter_7 = ExponentialMovingAverage(alpha=0.1)
    ema_filter_9 = ExponentialMovingAverage(alpha=0.1)
    ema_filter_11 = ExponentialMovingAverage(alpha=0.1)



    for joint_index in gripper_joints_index_list:
        p.setJointMotorControl2(robot_id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0)
    print("######所有手指关节设为速度模式")
    for _ in range(10000):
        p.stepSimulation()

        #计算软体质心速度
        soft_masscenter_now = calculate_soft_body_center_of_mass(soft_body_id)
        if soft_masscenter_pre != [0,0,0]:
            softbody_vel = [(soft_masscenter_now[i] - soft_masscenter_pre[i]) / time_step for i in range(3)]
            # print(f"软体的移动速度: {softbody_vel}")
        soft_masscenter_pre = soft_masscenter_now           

        # 获取接触点信息(可以用getContactPoints（indexA）简化！！！！)
        contact_points = p.getContactPoints(bodyA=robot_id)

        # 初始化接触点计数，明确统计链接6, 8, 10上的接触点数量
        contact_count_6810 = {6: 0, 8: 0, 10: 0}
        contact_count_7911 = {7: 0, 9: 0, 11: 0}

        # 遍历所有接触点，统计链接6、8、10的接触点数量
        for point in contact_points:
            link_index = point[3]  # 获取第一个物体的链接索引
            
            # 如果链接索引a或b是我们关心的链接，更新统计
            if link_index in contact_count_6810:
                contact_count_6810[link_index] += 1
                # print(contact_count)

        # 检查关节6、8、10的接触点数量是否都大于3
        if pass_6_8_10 == 0:
            count_link = sum(1 for joint in closing_joints if contact_count_6810[joint] >= 1)
            count_link_7_9_11 = sum(1 for joint in new_closing_joints if contact_count_7911[joint] >= 1)
            if count_link >= 2 or count_link_7_9_11 >=2:
                pass_6_8_10 = 1
                # 停止关节6、8、10的闭合
                print("6,8,10 stop!!!!")
                print("6,8,10 stop and hold position!!!!")
            # 如果条件不满足，继续闭合关节6、8、10
            else:
                print("######6810力控制")
                for joint in closing_joints:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint,
                        controlMode=p.TORQUE_CONTROL,
                        force=0.5  # 设置力矩，控制闭合力度
                    )



        #控制6，8，10关节不动
        if pass_6_8_10 ==1 and count_position == 0:
            for joint in closing_joints:
                p.setJointMotorControl2(robot_id, joint, controlMode=p.TORQUE_CONTROL, force=0)
            print("#####6810设为力控制111")
            
            current_position_6 = p.getJointState(robot_id, 6)[0]  # 获取当前关节位置
            current_position_8 = p.getJointState(robot_id, 8)[0]  # 获取当前关节位置
            current_position_10 = p.getJointState(robot_id, 10)[0]  # 获取当前关节位
            if count_position ==0:
                true_position_6 = current_position_6  #只有第一次循环到这里时的位置才是真的位置
                true_position_8 = current_position_8
                true_position_10 = current_position_10
                true_position = [true_position_6,true_position_8,true_position_10]
            count_position = 1  
            print('###6810位置控制111')
            for joint in closing_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=true_position[int(joint/2-3)],  # 目标位置为当前关节位置
                    force=0.5,  # 设置力以保持当前位置
                    maxVelocity=0.5
                    )

                
        #控制7，9，11关节
        if pass_6_8_10 == 1:    
            # 开始闭合关节7、9、11
            if cone_indicator == 0:
                print('######7910接触力控制')
                for joint in new_closing_joints:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint,
                        controlMode=p.TORQUE_CONTROL,
                        force=0.05  # 设置力矩，控制闭合力度
                    )
            if calculate_combined_friction_indicator(robot_id,[7,9,11],0.5,0.5) == True:
                # for joint in new_closing_joints:
                #     p.setJointMotorControl2(
                #         bodyUniqueId=robot_id,
                #         jointIndex=joint,
                #         controlMode=p.TORQUE_CONTROL,
                #         force=10  # 设置力矩，控制闭合力度
                #     )
                cone_indicator = 1

                #算滑动速度
                # vel_7 = np.array(list(p.getLinkState(robot_id, 7, computeLinkVelocity=1)[6]))
                # vel_9 = np.array(list(p.getLinkState(robot_id, 9, computeLinkVelocity=1)[6]))
                # vel_11 = np.array(list(p.getLinkState(robot_id, 11, computeLinkVelocity=1)[6]))
            if cone_indicator == 1:
                # gripper_vel = (vel_7+vel_9+vel_11)/3
                if gripper_close == 0:
                    vel_6 = np.array(list(p.getLinkState(robot_id, 6, computeLinkVelocity=1)[6]))
                    vel_8 = np.array(list(p.getLinkState(robot_id, 8, computeLinkVelocity=1)[6]))
                    vel_10 = np.array(list(p.getLinkState(robot_id, 10, computeLinkVelocity=1)[6]))
                    gripper_vel = (vel_6+vel_8+vel_10)/3

                    softbody_vel = np.array(softbody_vel)
                    slip_vel = softbody_vel - gripper_vel
                    # print(slip_vel)
                    slip_vel_z = slip_vel[2]

                    if frame_counter % frames_between_adjustments == 0:
                        # 使用PID计算力矩调整
                        # force_adjustment_ori = pid.compute(slip_vel_z, dt)
                        # print("slip_vel:",slip_vel_z,"------force_adj:",force_adjustment)
                        # force_adjustment = ema_filter.filter(force_adjustment_ori)
                        # print('force_adj:',force_adjustment)
                        force_adjustment = pid.compute(slip_vel_z, dt)
                        # 预测：如果力矩增加导致z速度加快，则减少力矩
                        if np.abs(slip_vel_z) > np.abs(previous_average_z_velocity) and (applied_force-previous_applied_force) > 0:
                            force_adjustment *= 0.8  # 缩小力矩
                        
                        # 如果z速度为零或软体物体向上移动，减少力矩
                        if slip_vel_z == 0 or slip_vel_z > 0:
                            force_adjustment *= 0.8  # 如果没有滑动或物体向上移动，减少力矩

                        # force_record=np.append(force_record,force_adjustment_ori)
                        # force_record_filter = np.append(force_record_filter,force_adjustment)
                        # time_force = np.append(time_force,frame_counter/frames_between_adjustments*dt)



                        # 确保力矩调整在最大允许范围内
                        if force_adjustment > max_torque:
                            force_adjustment = max_torque
                        elif force_adjustment < 0.01:
                            force_adjustment = 0.01

                        contact_forces = get_normal_forces(robot_id,link_indices=[7,9,11])
                        total_forces = contact_forces[7]+contact_forces[9]+contact_forces[11]
                        if total_forces == 0:
                            total_forces = 1
                        force_7 = contact_forces[7]
                        force_9 = contact_forces[9]
                        force_11 = contact_forces[11]
                        # print(f'force7:{force_7},force9:{force_9},force11:{force_11}')
                        max_contact_force = max((force_9+force_11),force_7)
                        min_contact_force = min((force_9+force_11),force_7)
                        if total_forces != 0:
                            if  force_9+force_11 != 0:  
                                if force_7 == max_contact_force:
                                    proportion_dic[7] = min_contact_force/total_forces
                                    proportion_dic[9] = 0.5*max_contact_force/total_forces
                                    proportion_dic[11] = 0.5*max_contact_force/total_forces
                                else:
                                    proportion_dic[7] = max_contact_force/total_forces
                                    proportion_dic[9] = 0.5*min_contact_force/total_forces
                                    proportion_dic[11] = 0.5*min_contact_force/total_forces
                            else:
                                proportion_dic[7] = 0.5
                                proportion_dic[9] = 0.25
                                proportion_dic[11] = 0.25
                                # print("force_9+force_11=0!")
                        else:
                            proportion_dic[7] = 0.5
                            proportion_dic[9] = 0.25
                            proportion_dic[11] = 0.25
                            # print("total_forces=0!")        

 

                        joint_state7 = p.getJointState(robot_id, 7)
                        joint_state9 = p.getJointState(robot_id, 9)
                        joint_state11 = p.getJointState(robot_id, 11)

                        # 提取旋转角度（位置）
                        joint_angle_7 = joint_state7[0]
                        joint_angle_9 = joint_state9[0]
                        joint_angle_11 = joint_state11[0] 

                        joint_angle_average =  (joint_angle_7+joint_angle_9+joint_angle_11)/3
                        
                        if (joint_angle_7+0.1)<joint_angle_average:
                            proportion_dic[7] = 0.8
                            proportion_dic[9] = 0.1
                            proportion_dic[11] = 0.1
                        if (joint_angle_9+0.1)<joint_angle_average and (joint_angle_11+0.1)<joint_angle_average:
                            proportion_dic[7] = 0.1
                            proportion_dic[9] = 0.45
                            proportion_dic[11] = 0.45
                        jointangle_dic[7] = joint_angle_7
                        jointangle_dic[9] = joint_angle_9
                        jointangle_dic[11] = joint_angle_11
                        if not (up_indicator == 1 and horizontal_indicator == 0):
                            if joint_angle_7>1.4:
                                proportion_dic[7] = 0.05
                            if joint_angle_9>1.4:
                                proportion_dic[9] = 0.05   
                            if joint_angle_11>1.4:
                                proportion_dic[11] = 0.05
                        if up_indicator == 1 and horizontal_indicator == 0:
                            if joint_angle_7>1.4:
                                proportion_dic[7] = proportion_dic[7]*0.5
                            if joint_angle_9>1.4:
                                proportion_dic[9] = proportion_dic[9]*0.5   
                            if joint_angle_11>1.4:
                                proportion_dic[11] = proportion_dic[11]*0.5

                        # print("Joint angle:", joint_angle_7,joint_angle_9,joint_angle_11)
                        # print('propotion:',proportion_dic[7],proportion_dic[9],proportion_dic[11])
                        # print("_______________________________")




                        # 对每个手指链接应用计算出的力矩
                        for link_index in [7,9,11]:
                            force_final = force_adjustment*proportion_dic[link_index]
                            if link_index == 7:
                                force_final = ema_filter_7.filter(force_final)
                            if link_index == 9:
                                force_final = ema_filter_9.filter(force_final)
                            if link_index == 11:
                                force_final = ema_filter_11.filter(force_final)
                            # print('force_final:',force_final)
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=link_index,
                                controlMode=p.TORQUE_CONTROL,
                                force=force_final
                                # force = 5
                            )
                        
                        # 更新上次应用的力矩
                        previous_applied_force = applied_force
                        applied_force = force_adjustment
                        
                        # 更新前一次的z速度，用于下一次迭代
                        previous_average_z_velocity = slip_vel_z

                        count_up_start += 1
                        contacts = p.getContactPoints(bodyA=robot_id,bodyB=soft_body_id)
                        for contact in contacts:
                            normal_force = contact[9]  # 获取法向力
                            total_contact_force += normal_force
                        # force_average = total_contact_force
                        if frame_counter % (frames_between_adjustments*10) == 0:
                            force_average = total_contact_force / (frames_between_adjustments*10)
                            total_contact_force = 0 
                            force_record=np.append(force_record,force_average)
                            time_force = np.append(time_force,frame_counter)

                            link_state_6 = p.getLinkState(robot_id, 6)
                            link_state_8 = p.getLinkState(robot_id, 8)
                            link_state_10 = p.getLinkState(robot_id, 10)

                            # 提取z方向的位置（第4个元素是位置，第2个值是z轴坐标）
                            z_position_6 = link_state_6[4][2]
                            z_position_8 = link_state_8[4][2]
                            z_position_10 = link_state_10[4][2]

                            # 计算z方向的平均位置
                            average_z_position_robot = (z_position_6 + z_position_8 + z_position_10) / 3
                            distance_record = np.append(distance_record,(average_z_position_robot-soft_masscenter_now[2]))
                            


                    #夹爪张开

                    #机械臂整体移动抬起
                    joint_positions = p.calculateInverseKinematics(robot_id, 5, [final_x_up, final_y_up, final_z_up], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                            
                    if up_indicator == 0:
                        active_joint_indices = list(range(6))  # 只控制 joint_1 到 joint_6
                        if count_up_start > 30:
                            #控制竖直上下运动
                            if calcu_height_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4]  # 第5个元素是世界坐标系下的位置
                                z_target_list = np.linspace(end_effector_position_5[2], final_z_up, num_height)
                                calcu_height_list = 1
                                position_target = z_target_list[count_height]

                            position_now = ((p.getLinkState(robot_id, 5))[4])[2]
                            
                            difference = position_now - position_target
                            
                            if  abs(difference) < 0.001:
                                count_height += 1

                                if count_height < num_height:
                                    position_target =  z_target_list[count_height] 
                                else:
                                    up_indicator = 1
                                    time.sleep(1)
                            joint_positions = p.calculateInverseKinematics(robot_id, 5, [final_x_up, final_y_up, position_target], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                            for j in range(6):
                                p.setJointMotorControl2(
                                    bodyUniqueId=robot_id,
                                    jointIndex=active_joint_indices[j],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[j],
                                    force=force_joint,
                                    maxVelocity=100)
                    if only_lift == False:
                        #控制水平移动
                        if up_indicator == 1 and horizontal_indicator == 0:
                            active_joint_indices = list(range(6))  # 只控制 joint_1 到 joint_6


                            if calcu_horizontal_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4]  # 第5个元素是世界坐标系下的位置
                                x_target_list = np.linspace(end_effector_position_5[0], final_x_horizontal, num_x_y)
                                y_target_list = np.linspace(end_effector_position_5[1], final_y_horizontal, num_x_y)
                                calcu_horizontal_list = 1
                                position_target_x = x_target_list[count_horizontal]
                                position_target_y = y_target_list[count_horizontal]


                            position_now_x = ((p.getLinkState(robot_id, 5))[4])[0]
                            position_now_y = ((p.getLinkState(robot_id, 5))[4])[1]

                            
                            difference = np.sqrt((position_target_x-position_now_x)**2+(position_target_y-position_now_y)**2)
                            
                            if  abs(difference) < 0.001:
                                count_horizontal += 1

                                if count_horizontal < num_x_y:
                                    position_target_x =  x_target_list[count_horizontal] 
                                    position_target_y =  y_target_list[count_horizontal] 

                                else:
                                    horizontal_indicator = 1
                                    time.sleep(1)
                            joint_positions = p.calculateInverseKinematics(robot_id, 5, [position_target_x, position_target_y, final_z_horizontal], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                            for j in range(6):
                                p.setJointMotorControl2(
                                    bodyUniqueId=robot_id,
                                    jointIndex=active_joint_indices[j],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[j],
                                    force=force_joint,
                                    maxVelocity=100)
                        
                        #控制下降
                        if up_indicator == 1 and horizontal_indicator == 1 and down_indicator ==0:
                            active_joint_indices = list(range(6))  # 只控制 joint_1 到 joint_6
                            if calcu_down_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4]  # 第5个元素是世界坐标系下的位置
                                z_target_list = np.linspace(end_effector_position_5[2], final_z_down, num_height)
                                calcu_down_list = 1
                                position_target = z_target_list[count_down]

                            position_now = ((p.getLinkState(robot_id, 5))[4])[2]
                            
                            difference = position_now - position_target
                            
                            if  abs(difference) < 0.001:
                                count_down += 1

                                if count_down < num_height:
                                    position_target =  z_target_list[count_down] 
                                else:
                                    down_indicator = 1
                                    time.sleep(1)
                                    gripper_close = 1 #夹爪张开
                            joint_positions = p.calculateInverseKinematics(robot_id, 5, [final_x_down, final_y_down, position_target], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                            for j in range(6):
                                p.setJointMotorControl2(
                                    bodyUniqueId=robot_id,
                                    jointIndex=active_joint_indices[j],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[j],
                                    force=force_joint,
                                    maxVelocity=100)

                else:
                    # print("夹爪张开！")
                    if gripper_6_8_10_refresh == 0:
                        for joint in [6,8,10]:
                            # p.resetJointState(robot_id, joint, targetValue=0)
                            p.setJointMotorControl2(robot_id, joint, controlMode=p.VELOCITY_CONTROL, force=0)
                        gripper_6_8_10_refresh = 1
                        print("######6810速度控制")
                    else:
                        for link_index in [6,7,8,9,10,11]:
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=link_index,
                                controlMode=p.TORQUE_CONTROL,
                                force = -0.02
                            )
                        count_finish += 1


                    if count_finish > 0.4/dt:
                        if all_finish_indicator ==0:
                            true_position_6 = p.getJointState(robot_id, 6)[0]  #只有第一次循环到这里时的位置才是真的位置
                            true_position_8 = p.getJointState(robot_id, 8)[0]
                            true_position_10 = p.getJointState(robot_id, 10)[0]
                            true_position_7 = p.getJointState(robot_id, 7)[0]
                            true_position_9 = p.getJointState(robot_id, 9)[0]
                            true_position_11 = p.getJointState(robot_id, 11)[0]

                            true_position = [true_position_6,true_position_7,true_position_8,
                                            true_position_9,true_position_10,true_position_11]
                            all_finish_indicator = 1  

                        print('力恒定！')
                        for joint in [6,7,8,9,10,11]:
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=true_position[joint-6],  # 目标位置为当前关节位置
                                force=10,  # 设置力以保持当前位置
                                maxVelocity=0.5
                                )
                        time.sleep(1)
                        break

                        


                        


        # for joint in [7,9,11]:
        #     joint_state = p.getJointState(robot_id, joint)
        #     applied_torque = joint_state[3]  # 第四个元素是当前施加的力矩（力）
    
        #     print(f"关节 {joint} 的当前施加力矩为: {applied_torque}")


        # 控制仿真速度
        p.setTimeStep(time_step)
        frame_counter +=1
    
    return time_force,force_record,force_record_filter,distance_record


def manipulator_move(robot_id ,target_position, max_force, desired_speed, end_effector_index=5,ran=500):
    for i in range(ran):
    # 获取机械臂的末端执行器位置
        end_effector_position = p.getLinkState(robot_id, end_effector_index)[4]

        # 计算目标位置和末端执行器位置之间的差距
        pos_diff = [target_position[i] - end_effector_position[i] for i in range(3)]

        # target_position[1] = target_position[1] - 0.01

        # 使用逆运动学计算各关节目标位置
        joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_position, targetOrientation=target_orientation)

        # 控制前6个关节（对应机械臂的主要关节）
        active_joint_indices = list(range(6))  # 只控制 joint_1 到 joint_6
        for j in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=active_joint_indices[j],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[j],
                force=max_force,
                maxVelocity=desired_speed)
        # 进行仿真步
        p.stepSimulation()
        time.sleep(1/1000)

def manipulator_vertical(robot_id, target_position_on_object, max_force, desired_speed,soft_body_id,end_effector_index=5,target_orientation = p.getQuaternionFromEuler([0, 0, 0]),
                         active_joint_indices = list(range(6)),ran=100):
    # target_position_on_object[1] = target_position_on_object[1] - 0.01
    for _ in range(ran):
        joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_position_on_object, target_orientation)
        for j in range(6):  # 控制机械臂前6个关节
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=active_joint_indices[j],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[j],
                force=max_force,
                maxVelocity=desired_speed)

    # 进行仿真步，确保机械臂完成下降动作
    # 你可以根据需要调整循环次数
        p.stepSimulation()
        time.sleep(1/1000)


# 定义输入量
target_position = [0.3-0.001, 0.5, 0.5]  # 抓取目标位置  ,减0.01
target_orientation = p.getQuaternionFromEuler([0, 0, 0])
target_position_on_object = [target_position[0], target_position[1], 0.214]
target_position_lift = [target_position[0], target_position[1], 0.5]
target_put_position = [0.4,-0.2,0.3]  #不用减0.01
desired_speed = 10  # 调整这个值以控制速度
max_force = 200  # 调整这个值以控制力
gripper_joint_indices = [6,7, 8, 9, 10, 11]
torque = 50

soft_body_id = p.loadSoftBody(
    # r"ros_kortex\object\cube_mesh.vtk",  # 使用PyBullet自带的网格文件生成一个球状软体
    r"ros_kortex\object\ball.vtk",
    # r"ros_kortex\object\tube.vtk",
    basePosition=[0.3-0.0005, 0.5, 0.1],
    # baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
    scale=0.045,
    mass=0.3,
    collisionMargin=0.0001,
    useNeoHookean=True,
    useMassSpring=False,
    NeoHookeanMu=50000, 
    NeoHookeanLambda=100000, 
    NeoHookeanDamping=0.1,
    frictionCoeff=0.7,
    useFaceContact=True,
    useSelfCollision=False
)
# 创建篮子（使用盒子形状的凹形结构来模拟）
basket_position = [0.4, -0.2, 0.1]
basket_orientation = p.getQuaternionFromEuler([0, 0, 0])
basket_half_extents = [0.2, 0.2, 0.1]  # 设置篮子的尺寸

# 创建篮子的底部
basket_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[basket_half_extents[0], basket_half_extents[1], 0.02])
basket_base_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_base, basePosition=[basket_position[0], basket_position[1], basket_position[2] - basket_half_extents[2]])

# 创建篮子的侧面
basket_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[basket_half_extents[0], 0.02, basket_half_extents[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0], basket_position[1] + basket_half_extents[1], basket_position[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0], basket_position[1] - basket_half_extents[1], basket_position[2]])

basket_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, basket_half_extents[1], basket_half_extents[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0] + basket_half_extents[0], basket_position[1], basket_position[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0] - basket_half_extents[0], basket_position[1], basket_position[2]])
# soft_body_id_2 = p.loadSoftBody(
#     r"ros_kortex\object\ball.vtk",  # 使用PyBullet自带的网格文件生成一个球状软体
#     basePosition=[0.2, 0.5, 0.1],
#     scale=0.05,
#     mass=1,
#     collisionMargin=0.001,
#     useNeoHookean=False,
#     useMassSpring=True,
#     useBendingSprings=True,
#     springElasticStiffness=100,
#     springDampingStiffness=100,
#     frictionCoeff=5,
#     useFaceContact=True,
#     useSelfCollision=False
# )
# aabb_min, aabb_max = p.getAABB(soft_body_id)
# 获取软体的轴对齐边界框（AABB）
aabb_min, aabb_max = p.getAABB(soft_body_id)

# 计算并打印软体的高度（Z轴范围）
soft_body_height = aabb_max[2] - aabb_min[2]
print("软体的最小高度:", aabb_min[2])
print("软体的最大高度:", aabb_max[2])
print("软体的高度:", soft_body_height)
# # 计算软体物体的尺寸
# size_x = aabb_max[0] - aabb_min[0]
# size_y = aabb_max[1] - aabb_min[1]
# size_z = aabb_max[2] - aabb_min[2]
# print("###################################")
# print(f'x:{size_x},y:{size_y},z:{aabb_max}')
# print("###################################")
p.setPhysicsEngineParameter(enableConeFriction=1)
# get_joint_info(robot_id)
# manipulator_move(robot_id,target_position,max_force,desired_speed)
# manipulator_vertical(robot_id,target_position_on_object,max_force,desired_speed)
# gripper_control(robot_id,gripper_joint_indices,torque)
# manipulator_vertical(robot_id,target_position_lift,max_force,desired_speed)

manipulator_move(robot_id,target_position,max_force,desired_speed,ran=50)
manipulator_vertical(robot_id,target_position_on_object,max_force,desired_speed,soft_body_id=soft_body_id,ran=50)
start_time = time.time()
time_force,force_record,force_record_filter,distance_record = gripper_control(robot_id,grasp_position=target_position_on_object,put_position=target_put_position,only_lift=False)
# print('!!!!!!!!!!!!!!!!!!!!!!!!!')
end_time = time.time()
# 计算真实时间差
real_time_elapsed = end_time - start_time

print("Total real time elapsed:", real_time_elapsed, "seconds")
# manipulator_move(robot_id,[0.2,0.5-0.01,0.5],max_force,desired_speed,ran=500)
# manipulator_vertical(robot_id,[0.2,0.5-0.01,0.22],max_force,desired_speed,soft_body_id=soft_body_id,ran=500)

# gripper_control(robot_id,grasp_position=[0.2,0.5-0.01,0.22],put_position=target_put_position)
np.savetxt('ros_kortex/data_record/time_step.csv', time_force, delimiter=',', fmt='%f')
np.savetxt('ros_kortex/data_record/box_force.csv', force_record, delimiter=',', fmt='%f')
np.savetxt('ros_kortex/data_record/box_distance.csv', distance_record, delimiter=',', fmt='%f')

