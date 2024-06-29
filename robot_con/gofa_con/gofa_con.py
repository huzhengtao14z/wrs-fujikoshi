"""
Created on 2024/6/14 
Author: Hao Chen (chen960216@gmail.com)
"""
import time
import numpy as np

from .gofa_arm import GoFaArm
from .gofa_state import GoFaState
from .piecewisepoly import PiecewisePoly
from .gofa_constants import GoFaConstants as GFC


class GoFaArmController:
    """
    A client to control the yumi
    """

    def __init__(self, toggle_debug=False, toggle_monitor_only=False):
        self._toggle_monitor_only = toggle_monitor_only
        if not toggle_monitor_only:
            self.rbtx = GoFaArm(debug=toggle_debug)
            self._is_add_all = True
            self._traj_opt = PiecewisePoly()
        self.sec_rbtx = GoFaArm(debug=toggle_debug, port=GFC.PORTS['states'])

    @property
    def arm(self):
        return self.rbtx if not self._toggle_monitor_only else self.sec_rbtx

    def get_pose(self, component_name, return_conf=False):
        raise NotImplementedError

    def get_jnt_values(self, ):
        """
        get the joint angles of both arms
        :return: 1x6 array
        author: chen
        """
        return np.deg2rad(self.sec_rbtx.get_state().joints)

    def get_torques(self) -> np.ndarray:
        """
        get the torques of joints
        See 1: https://library.e.abb.com/public/b227fcd260204c4dbeb8a58f8002fe64/Rapid_instructions.pdf
        See 2: https://forums.robotstudio.com/discussion/13247/motor-torque-using-getmotortorque-vs-getjointdata
        Notes: When the GoFa joints lock (idle), the torques are zeros.
        :return: joints
        """
        return np.asarray(self.sec_rbtx.get_torques())

    def get_torques_current(self) -> np.ndarray:
        """
        get the torques of joints
        See 1: https://library.e.abb.com/public/b227fcd260204c4dbeb8a58f8002fe64/Rapid_instructions.pdf
        See 2: https://forums.robotstudio.com/discussion/13247/motor-torque-using-getmotortorque-vs-getjointdata
        Notes: When the GoFa joints lock (idle), the torques are zeros.
        :return: joints
        """
        return np.asarray(self.sec_rbtx.get_torques_current())

    def move_j(self, jnt_vals: np.ndarray, speed_n=100, wait=True):
        """
        move one arm joints of the yumi
        :param component_name
        :param jnt_vals: 1x7 np.array
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        :return: bool

        author: weiwei
        date: 20170411
        """
        if self._toggle_monitor_only:
            raise Exception("Toggle off monitor only to enable robot movements")
        assert len(jnt_vals) == GoFaState.NUM_JOINTS
        if speed_n == -1:
            self.arm.set_speed_max()
        else:
            speed_data = self.rbtx.get_v(speed_n)
            self.arm.set_speed(speed_data)

        armjnts = np.rad2deg(jnt_vals)
        ajstate = GoFaState(armjnts)
        self.arm.movetstate_sgl(ajstate, wait_for_res=wait)

    def fk(self, component_name: str, jnt_vals: np.ndarray, return_conf: bool = False) -> tuple:
        raise NotImplementedError

    def ik(self, component_name: str,
           pos: np.ndarray,
           rot: np.ndarray,
           conf: np.ndarray = None,
           ext_axis: float = None) -> np.ndarray or None:
        raise NotImplementedError

    def move_jntspace_path(self, path, speed_n=100, wait=True) -> bool:
        """
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        """
        if self._toggle_monitor_only:
            raise Exception("Toggle off monitor only to enable robot movements")
        statelist = []
        st = time.time()
        for armjnts in self._traj_opt.interpolate_path(path, num=min(100, int(len(path)))):
            armjnts = np.rad2deg(armjnts)
            ajstate = GoFaState(armjnts)
            statelist.append(ajstate)
        et = time.time()
        print("time calculating sending information", et - st)
        # set the speed of the robot
        if speed_n == -1:
            self.arm.set_speed_max()
        else:
            speed_data = self.arm.get_v(speed_n)
            self.arm.set_speed(speed_data)
        exec_result = self.arm.movetstate_cont(statelist, is_add_all=self._is_add_all, wait_for_res=wait)
        return exec_result

    def stop(self):
        if not self._toggle_monitor_only:
            self.rbtx.stop()
        self.sec_rbtx.stop()


if __name__ == "__main__":
    yumi_con = GoFaArmController()
    print(yumi_con.get_jnt_values())
