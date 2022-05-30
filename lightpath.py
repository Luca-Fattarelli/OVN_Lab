import signal_information
import line_constants


class Lightpath(signal_information.Signal_information):

    def __init__(self, sig_power_value: float, path: list, att_channel: int):
        super(Lightpath, self).__init__(sig_power_value, path)
        self.attribute_channel = att_channel
        self.r_s = line_constants.r_s
        self.df = line_constants.df

    # TODO: setters and getters

    def get_channel(self):
        return self.attribute_channel
