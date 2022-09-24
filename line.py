import scipy.constants

import line_constants
import node
import signal_information
import lightpath
import math_db
import numpy as np


class Line:

    def __init__(self, label: str, length: float):
        self.label = label
        self.length = length
        self.successive = {}
        self.state = []
        self.n_amplifiers = length / 80000 + 2
        self.number_of_channels = 1
        self.opt_launch_power = 0
        self.in_service = 1
        self.nli_d = None

    # TODO: getter and setter

    def setup_state(self, number_of_channels):
        self.number_of_channels = number_of_channels
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
        self.state[channel_index] = state

    def get_state(self, channel_index):
        return self.state[channel_index]

    def latency_generation(self, sig_info: signal_information.Signal_information):
        transmission_speed = 2 * 10 ** 8
        lat = self.length / transmission_speed
        return lat

    def noise_generation(self, sig_info: signal_information.Signal_information):

            # OLD NOISE DEFINITION
        # noise = 1e-9 * sig_info.get_signal_power() * self.length

        # LAB8 NOISE DEFINITION
        noise = self.ase_generation() + self.nli_generation(self.number_of_channels, sig_info.get_signal_power())
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

    # returns Amplified Spontaneous Emissions in LINEAR UNITS
    def ase_generation(self):
        res = self.n_amplifiers * (scipy.constants.Planck * line_constants.f * line_constants.b_n *
                                   math_db.db_to_linear(line_constants.noise_figure) *
                                   (math_db.db_to_linear(line_constants.gain) - 1))
        return res

    def nli_d_generation(self, n_span):
        self.nli_d = (16 / 27 / np.pi) * (line_constants.gamma ** 2) / \
                (4 * line_constants.beta_2 * line_constants.alfa * (line_constants.r_s ** 3)) * \
                np.log10((np.pi ** 2) / 2 * line_constants.beta_2 * (line_constants.r_s ** 2) /
                         line_constants.alfa * (n_span ** (2 * line_constants.r_s /
                                                           line_constants.df)))
        return self.nli_d

    def nli_generation(self, n_span, channel_power):
        nli = line_constants.b_n * n_span * (channel_power * channel_power * channel_power) * \
              self.nli_d_generation(n_span)
        return nli

    def optimized_launch_power(self):
        # std formula
        # p_opt = line_constants.f * self.length * p_base / (2 * channel_bandwidth * self.nli_d)
        p_opt = self.ase_generation() / 2 / self.nli_d_generation(self.number_of_channels)
        return p_opt ** (1/3)

    def set_opt_launch_power(self, p_opt):
        self.opt_launch_power = p_opt

    def get_in_service(self):
        return self.in_service

    def set_in_service(self, val):
        self.in_service = val
