"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
if __name__ == "__main__":
    import time

    from gofa_con import GoFaArmController

    # initialize the GoFaArm Controller
    arm = GoFaArmController(toggle_debug=False)
    for i in range(1000):
        torque_val = arm.get_torques()
        print("Torque values: ", torque_val)
        time.sleep(.1)
    arm.stop()