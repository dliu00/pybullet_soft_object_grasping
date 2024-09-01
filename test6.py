import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

p.connect(p.GUI) # Start the graphical user interface
# physicsClient = p.connect(p.DIRECT) # Do not start the graphical user interface

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
camera_distance = 1.0  # Distance between the visualization interface and the target point
camera_yaw = 80        # Horizontal rotation angle of the visualization interface (centered on the target point)
camera_pitch = -10     # Pitch angle of the visualization interface
camera_target_position = [0.3, 0.5-0.01,0]  # Target point that the visualization interface is aimed at (usually the robot's position)

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=5)
robot_id = p.loadURDF(r"ros_kortex\myrobo.urdf", useFixedBase=True)

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

    mu = min(object_friction, gripper_friction) 
    total_normal_force = 0
    total_tangential_force = np.array([0.0, 0.0, 0.0])  
    
    for link_index in link_indices:
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)
        
        for contact in contact_points:
            normal_force = contact[9]  
            lateral_friction1 = contact[10]  
            lateral_friction_dir1 = np.array(contact[11]) 
            lateral_friction2 = contact[12]  
            lateral_friction_dir2 = np.array(contact[13])  
            
            # Synthesize tangential friction force
            lateral_force_total = (lateral_friction1 * lateral_friction_dir1 +
                                   lateral_friction2 * lateral_friction_dir2)

            total_normal_force += normal_force
            total_tangential_force += lateral_force_total
    

    total_tangential_magnitude = np.linalg.norm(total_tangential_force)
    
    is_within_friction_cone = total_tangential_magnitude <= mu * total_normal_force

    return is_within_friction_cone

def get_normal_forces(robot_id, link_indices=[7,9,11]):
   
    average_forces = {7: 0, 9: 0, 11: 0}
    
    for link_index in link_indices:
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)
        
        for point in contact_points:
            average_forces[link_index] += point[9] 
 
    return average_forces

def calculate_average_contact_velocity(robot_id, finger_links):
    
    average_velocities = {}

    for link_index in finger_links:
        contact_points = p.getContactPoints(bodyA=robot_id, linkIndexA=link_index)

        if len(contact_points) == 0:
            average_velocities[link_index] = np.array([0.0, 0.0, 0.0])
            continue

        total_velocity = np.array([0.0, 0.0, 0.0])
        count = 0
        for point in contact_points:
            relative_velocity = np.array(point[11])  
            total_velocity += relative_velocity
            count += 1
        average_velocity = total_velocity / count if count > 0 else np.array([0.0, 0.0, 0.0])
        average_velocities[link_index] = average_velocity

    return average_velocities

def calculate_soft_body_center_of_mass(soft_body_id):
  
    mesh_data = p.getMeshData(soft_body_id, flags=p.MESH_DATA_SIMULATION_MESH)
    node_positions = mesh_data[1]
    
    # Ensure that node data is not empty
    num_nodes = len(node_positions)
    if num_nodes == 0:
        raise ValueError("No nodes found in the soft body.")
    center_of_mass = [sum(pos[i] for pos in node_positions) / num_nodes for i in range(3)] 
    return center_of_mass

def gripper_control(robot_id, grasp_position, put_position, only_lift, lift_height=0.4, pid_params=(2,0.001,0.03), max_torque=10, mode=p.TORQUE_CONTROL,ran=100,time_step=1/1000):
    closing_joints = [6, 8, 10]
    new_closing_joints = [7,9,11]
    gripper_joints_index_list = [6,7,8,9,10,11]
    print("Gripper begin!!!!")
    pass_6_8_10 = 0
    soft_masscenter_now = [0,0,0]
    soft_masscenter_pre = [0,0,0]
    count_position = 0  # Determine if it's in the holding position
    softbody_vel = 0
    cone_indicator = 0  # Determine if the friction cone condition is met
    gripper_close = 0 # Gripper closure status
    gripper_6_8_10_refresh = 0 # Switch from position control to torque control
    proportion_dic = {7:0.0,9:0.0,11:0.0}  # Control proportions for the forces on the three fingers
    jointangle_dic = {7:0.0,9:0.0,11:0.0}

    force_joint = 100

    frame_counter = 0
    frames_between_adjustments = 2
    previous_average_z_velocity = 0.0
    applied_force = 0.0 # Record the last applied force
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

    calcu_height_list = 0  
    calcu_horizontal_list = 0
    calcu_down_list = 0

    count_height = 0  
    count_horizontal = 0
    count_down = 0
    count_finish = 0
    count_up_start = 0

    num_height = 100  # Number of segments in the vertical motion path
    num_x_y = 100

    up_indicator = 0 # Determine if the upward motion is complete
    down_indicator = 0 # Determine if the downward motion is complete
    horizontal_indicator = 0 # Determine if the horizontal motion is complete
    all_finish_indicator = 0
    total_contact_force = 0 

    ema_filter_7 = ExponentialMovingAverage(alpha=0.1)
    ema_filter_9 = ExponentialMovingAverage(alpha=0.1)
    ema_filter_11 = ExponentialMovingAverage(alpha=0.1)

    for joint_index in gripper_joints_index_list:
        p.setJointMotorControl2(robot_id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0)
    for _ in range(10000):
        p.stepSimulation()

        soft_masscenter_now = calculate_soft_body_center_of_mass(soft_body_id)
        if soft_masscenter_pre != [0,0,0]:
            softbody_vel = [(soft_masscenter_now[i] - soft_masscenter_pre[i]) / time_step for i in range(3)]
        soft_masscenter_pre = soft_masscenter_now           

        contact_points = p.getContactPoints(bodyA=robot_id)

        contact_count_6810 = {6: 0, 8: 0, 10: 0}
        contact_count_7911 = {7: 0, 9: 0, 11: 0}

        # Traverse all contact points and count the number of contacts for links 6, 8, and 10
        for point in contact_points:
            link_index = point[3] 

            if link_index in contact_count_6810:
                contact_count_6810[link_index] += 1
                # print(contact_count)

        # Check if the number of contact points on links 6, 8, and 10 meets the conditions
        if pass_6_8_10 == 0:
            count_link = sum(1 for joint in closing_joints if contact_count_6810[joint] >= 1)
            count_link_7_9_11 = sum(1 for joint in new_closing_joints if contact_count_7911[joint] >= 1)
            if count_link >= 2 or count_link_7_9_11 >=2:
                pass_6_8_10 = 1
                print("6,8,10 stop!!!!")
                print("6,8,10 stop and hold position!!!!")
            # If the conditions are not met, continue closing joints 6, 8, and 10
            else:
                for joint in closing_joints:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint,
                        controlMode=p.TORQUE_CONTROL,
                        force=0.5  
                    )

        if pass_6_8_10 ==1 and count_position == 0:
            for joint in closing_joints:
                p.setJointMotorControl2(robot_id, joint, controlMode=p.TORQUE_CONTROL, force=0)
            
            current_position_6 = p.getJointState(robot_id, 6)[0]  
            current_position_8 = p.getJointState(robot_id, 8)[0]  
            current_position_10 = p.getJointState(robot_id, 10)[0]  
            if count_position ==0:
                true_position_6 = current_position_6  # The first time the loop reaches here is the true position
                true_position_8 = current_position_8
                true_position_10 = current_position_10
                true_position = [true_position_6,true_position_8,true_position_10]
            count_position = 1  

            for joint in closing_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=true_position[int(joint/2-3)], 
                    force=0.5,  
                    maxVelocity=0.5
                    )

        # Control joints 7, 9, and 11
        if pass_6_8_10 == 1:    
            if cone_indicator == 0:
                for joint in new_closing_joints:
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint,
                        controlMode=p.TORQUE_CONTROL,
                        force=0.05 
                    )
            if calculate_combined_friction_indicator(robot_id,[7,9,11],0.5,0.5) == True:
                cone_indicator = 1

            if cone_indicator == 1:
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

                        force_adjustment = pid.compute(slip_vel_z, dt)
                        # Prediction: If increasing the torque results in an increase in z-velocity, reduce the torque
                        if np.abs(slip_vel_z) > np.abs(previous_average_z_velocity) and (applied_force-previous_applied_force) > 0:
                            force_adjustment *= 0.8  
                        
                        # If z-velocity is zero or the soft body is moving upward, reduce the torque
                        if slip_vel_z == 0 or slip_vel_z > 0:
                            force_adjustment *= 0.8 

                        # Ensure the torque adjustment is within the maximum allowed range
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

                        # Extract rotation angles (positions)
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

                        # Apply the calculated torque to each finger link
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
                            )
                        
                        # Update the last applied torque
                        previous_applied_force = applied_force
                        applied_force = force_adjustment
                        
                        # Update the previous z-velocity for the next iteration
                        previous_average_z_velocity = slip_vel_z

                        count_up_start += 1
                        contacts = p.getContactPoints(bodyA=robot_id,bodyB=soft_body_id)
                        for contact in contacts:
                            normal_force = contact[9]  
                            total_contact_force += normal_force
                        if frame_counter % (frames_between_adjustments*10) == 0:
                            force_average = total_contact_force / (frames_between_adjustments*10)
                            total_contact_force = 0 
                            force_record=np.append(force_record,force_average)
                            time_force = np.append(time_force,frame_counter)

                            link_state_6 = p.getLinkState(robot_id, 6)
                            link_state_8 = p.getLinkState(robot_id, 8)
                            link_state_10 = p.getLinkState(robot_id, 10)
                            z_position_6 = link_state_6[4][2]
                            z_position_8 = link_state_8[4][2]
                            z_position_10 = link_state_10[4][2]

                            average_z_position_robot = (z_position_6 + z_position_8 + z_position_10) / 3
                            distance_record = np.append(distance_record,(average_z_position_robot-soft_masscenter_now[2]))
                            
                    joint_positions = p.calculateInverseKinematics(robot_id, 5, [final_x_up, final_y_up, final_z_up], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                            
                    if up_indicator == 0:
                        active_joint_indices = list(range(6))  
                        if count_up_start > 30:
                            # Control vertical motion
                            if calcu_height_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4] 
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
                        # Control horizontal movement
                        if up_indicator == 1 and horizontal_indicator == 0:
                            active_joint_indices = list(range(6))  

                            if calcu_horizontal_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4]  
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
                        
                        # Control descent
                        if up_indicator == 1 and horizontal_indicator == 1 and down_indicator ==0:
                            active_joint_indices = list(range(6)) 
                            if calcu_down_list == 0:
                                link_state_5 = p.getLinkState(robot_id, 5)
                                end_effector_position_5 = link_state_5[4]  
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
                                    gripper_close = 1 # Gripper opens
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
                    if gripper_6_8_10_refresh == 0:
                        for joint in [6,8,10]:
                            p.setJointMotorControl2(robot_id, joint, controlMode=p.VELOCITY_CONTROL, force=0)
                        gripper_6_8_10_refresh = 1
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
                            true_position_6 = p.getJointState(robot_id, 6)[0]  # The first time the loop reaches here is the true position
                            true_position_8 = p.getJointState(robot_id, 8)[0]
                            true_position_10 = p.getJointState(robot_id, 10)[0]
                            true_position_7 = p.getJointState(robot_id, 7)[0]
                            true_position_9 = p.getJointState(robot_id, 9)[0]
                            true_position_11 = p.getJointState(robot_id, 11)[0]

                            true_position = [true_position_6,true_position_7,true_position_8,
                                            true_position_9,true_position_10,true_position_11]
                            all_finish_indicator = 1  

                        for joint in [6,7,8,9,10,11]:
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=true_position[joint-6],  
                                force=10, 
                                maxVelocity=0.5
                                )
                        time.sleep(1)
                        break
                    
        p.setTimeStep(time_step)
        frame_counter +=1
    
    return time_force,force_record,force_record_filter,distance_record


def manipulator_move(robot_id ,target_position, max_force, desired_speed, end_effector_index=5,ran=500):
    for i in range(ran):
        end_effector_position = p.getLinkState(robot_id, end_effector_index)[4]
        pos_diff = [target_position[i] - end_effector_position[i] for i in range(3)]

        joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_position, targetOrientation=target_orientation)

        active_joint_indices = list(range(6))  
        for j in range(6):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=active_joint_indices[j],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[j],
                force=max_force,
                maxVelocity=desired_speed)
        p.stepSimulation()
        time.sleep(1/1000)

def manipulator_vertical(robot_id, target_position_on_object, max_force, desired_speed,soft_body_id,end_effector_index=5,target_orientation = p.getQuaternionFromEuler([0, 0, 0]),
                         active_joint_indices = list(range(6)),ran=100):
    for _ in range(ran):
        joint_positions = p.calculateInverseKinematics(robot_id, end_effector_index, target_position_on_object, target_orientation)
        for j in range(6):  
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=active_joint_indices[j],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[j],
                force=max_force,
                maxVelocity=desired_speed)

        p.stepSimulation()
        time.sleep(1/1000)


# Define input variables
target_position = [0.3-0.001, 0.5, 0.5]  # Grasp target position, subtract 0.001 to correct error
target_orientation = p.getQuaternionFromEuler([0, 0, 0])
target_position_on_object = [target_position[0], target_position[1], 0.214]
target_position_lift = [target_position[0], target_position[1], 0.5]
target_put_position = [0.3,0,0.25]  # No need to subtract 0.01
desired_speed = 10  
max_force = 200 
gripper_joint_indices = [6,7, 8, 9, 10, 11]
torque = 50

soft_body_id = p.loadSoftBody(
    r"ros_kortex\object\ball.vtk",
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
# Create a box
# Create a basket
basket_position = [0.3, 0, 0.05]
basket_orientation = p.getQuaternionFromEuler([0, 0, 0])
basket_half_extents = [0.1, 0.1, 0.05]  # Set basket dimensions

# Create the bottom of the basket
basket_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[basket_half_extents[0], basket_half_extents[1], 0.02])
basket_base_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_base, basePosition=[basket_position[0], basket_position[1], basket_position[2] - basket_half_extents[2]])

basket_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[basket_half_extents[0], 0.02, basket_half_extents[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0], basket_position[1] + basket_half_extents[1], basket_position[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0], basket_position[1] - basket_half_extents[1], basket_position[2]])

basket_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, basket_half_extents[1], basket_half_extents[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0] + basket_half_extents[0], basket_position[1], basket_position[2]])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=basket_side, basePosition=[basket_position[0] - basket_half_extents[0], basket_position[1], basket_position[2]])

manipulator_move(robot_id,target_position,max_force,desired_speed,ran=50)
manipulator_vertical(robot_id,target_position_on_object,max_force,desired_speed,soft_body_id=soft_body_id,ran=50)
start_time = time.time()
time_force,force_record,force_record_filter,distance_record = gripper_control(robot_id,grasp_position=target_position_on_object,put_position=target_put_position,only_lift=False)
# print('!!!!!!!!!!!!!!!!!!!!!!!!!')
end_time = time.time()
real_time_elapsed = end_time - start_time
