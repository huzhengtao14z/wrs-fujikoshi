import drivers.devices.dh.dh_modbus_gripper as dh_modbus_gripper
from time import sleep


class Ag145driver():
    def __init__(self, port = 'com3', baudrate = 115200, force = 100, speed = 100):
        port = port
        baudrate = baudrate
        initstate = 0
        # g_state = 0
        force = force
        speed = speed
        self.m_gripper = dh_modbus_gripper.dh_modbus_gripper()
        self.m_gripper.open(port, baudrate)
        self.init_gripper()
        while (initstate != 1):
            initstate = self.m_gripper.GetInitState()
            sleep(0.2)
        self.m_gripper.SetTargetPosition(500)
        self.set_speed(speed)
        self.set_force(force)

    def init_gripper(self):
        self.m_gripper.Initialization()
        print('Send grip init')

    def set_force(self, force=100):
        self.m_gripper.SetTargetForce(force)

    def set_speed(self, speed=100):
        self.m_gripper.SetTargetSpeed(speed)

    def conv2encoder(self, jawwidth):
        a = int(jawwidth * 1000 / 0.145)
        return a

    def jaw_to(self, jawwidth):
        self.m_gripper.SetTargetPosition(self.conv2encoder(jawwidth))
        g_state = 0
        while (g_state == 0):
            g_state = self.m_gripper.GetGripState()
            sleep(0.2)

    def open_g(self):
        self.jaw_to(0.145)

    def close_g(self):
        self.jaw_to(0)
