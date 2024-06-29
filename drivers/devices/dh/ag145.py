import drivers.devices.dh.dh_modbus_gripper as dh_modbus_gripper
import drivers.devices.dh.dh_socket_gripper as dh_socket_gripper
from time import sleep


class Ag145driver():
    def __init__(self):
        port = 'com3'
        baudrate = 115200
        initstate = 0
        g_state = 0
        force = 100
        speed = 100
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)


        self.m_gripper.Initialization()

        # self.init_gripper()

        # self.set_speed()
        # self.set_force()

    def init_gripper(self):
        self.m_gripper.Initialization()
        initstate = 0
        while (initstate != 1):
            initstate = self.m_gripper.GetInitState()
            sleep(0.2)
        print('Send grip init')

    def set_force(self, force=100):
        self.m_gripper.SetTargetForce(force)

    def set_speed(self, speed=100):
        self.m_gripper.SetTargetSpeed(speed)

    def conv2encoder(self, jawwidth):

        return int(jawwidth*1000/0.145)

    def jaw_to(self, jawwidth):
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state != 1):
            g_state = self.m_gripper.GetInitState()
            sleep(0.2)

    def open_g(self):
        self.jaw_to(0.145)

    def close_g(self):
        self.jaw_to(0)
