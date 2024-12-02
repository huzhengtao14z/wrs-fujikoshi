"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
if __name__ == "__main__":
    import math
    from robot_con.gofa_con import GoFaArmController
    paths = []
    # initialize the GoFaArm Controller
    arm = GoFaArmController(toggle_debug=False)
    if len(paths) > 0:
        arm.move_j(paths[0])
        arm.move_jntspace_path(paths, speed_n=100)
    arm.stop()
