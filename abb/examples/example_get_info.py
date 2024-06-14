"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
if __name__ == "__main__":
    from robot_con.gofa_con import GoFaArmController

    # initialize the GoFaArm Controller
    arm = GoFaArmController(toggle_debug=False)
    j_val = arm.get_jnt_values()
    print("Joint values: ", j_val)
    arm.stop()