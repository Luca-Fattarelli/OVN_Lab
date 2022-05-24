import node
import signal_information
import lightpath
import numpy as np


class Line:

    def __init__(self, label: str, length: float):
        self.label = label
        self.length = length
        self.successive = {}
        self.state = []

    # TODO: getter and setter

    def setup_state(self, number_of_channels):
        for i in range(0, number_of_channels):
            self.state.append("1")

    def set_successive(self, label: str, node1: node.Node):
        self.successive[label] = node1

    def get_all_states(self):
        return self.state

    def decrease_state(self):
        self.state[0] = 0

    def free_state(self):
        self.state = [1]
    # speed m /s

    def set_state(self, channel_index, state):
        self.state[channel_index] = 0

    def get_state(self, channel_index):
        return self.state[channel_index]

    def latency_generation(self,sig_info: signal_information.Signal_information):
        transmission_speed = 2 * 10 ** 8
        lat = self.length / transmission_speed
        return lat

    def noise_generation(self, sig_info: signal_information.Signal_information):
        #noise = np.exp(-9) * sig_info.get_signal_power() * self.length
        noise = 1e-9 * sig_info.get_signal_power() * self.length
        return noise

    def propagate(self, sig: lightpath.Lightpath):
        sig.noise_power_increase(self.noise_generation(sig))
        sig.latency_increase(self.latency_generation(sig))
        node1 = self.successive.get(self.label[1])
        self.set_state(sig.get_channel(), 0)
        node1.propagate(sig)

    def probe(self, sig_info: lightpath.Lightpath):
        sig_info.noise_power_increase(self.noise_generation(sig_info))
        sig_info.latency_increase(self.latency_generation(sig_info))
        node1 = self.successive.get(self.label[1])
        # node propagation
        node1.probe(sig_info)


