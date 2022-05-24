class Connection:
    def __init__(self, input: str, output: str, signal_power: float):
        self.input = input
        self.output = output
        self.signal_power = signal_power
        self.latency = 0
        self.snr = 0
        self.bit_rate = 0

    def set_bit_rate(self, bit_rate):
        self.bit_rate = bit_rate

    def get_bit_rate(self):
        return self.bit_rate

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_signal_power(self):
        return self.signal_power

    def set_latency(self, lat):
        self.latency = lat

    def set_snr(self, snr):
        self.snr = snr

    def get_lat(self):
        return self.latency

    def get_snr(self):
        return self.snr
