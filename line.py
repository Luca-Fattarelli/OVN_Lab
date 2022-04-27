import node
import signal_information
import numpy as np


class Line:

    def __init__(self, label: str, length: float):
        self.label = label
        self.length = length
        self.successive = {}
        self.state = 1

    # TODO: getter and setter

    def set_successive(self, label: str, node1: node.Node):
        self.successive[label] = node1

    def get_state(self):
        return self.state

    def decrease_state(self):
        self.state -= 1

    def free_state(self):
        self.state = 1
    # speed m /s

    def latency_generation(self,sig_info: signal_information.Signal_information):
        transmission_speed = 2 * 10 ^ 8
        lat = self.length / transmission_speed
        return lat

    def noise_generation(self, sig_info: signal_information.Signal_information):
        noise = np.exp(-9) * sig_info.get_signal_power() * self.length
        return noise

    def propagate(self, sig_info: signal_information.Signal_information):
        sig_info.noise_power_increase(self.noise_generation(sig_info))
        sig_info.latency_increase(self.latency_generation(sig_info))
        node1 = self.successive.get(self.label[1])
        # node propagation
        node1.propagate(sig_info)


