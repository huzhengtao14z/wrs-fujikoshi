"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
Rotate the last joint of the GoFa Arm to pi/2
"""
if __name__ == "__main__":
    import math
    from robot_con.gofa_con import GoFaArmController

    # initialize the GoFaArm Controller
    arm = GoFaArmController(toggle_debug=False)
    j_val = arm.get_jnt_values()
    print("Joint values: ", j_val)
    j_val_move = j_val.copy()
    j_val_move[-1] = math.pi / 2
    arm.move_j(j_val_move)
    arm.stop()
