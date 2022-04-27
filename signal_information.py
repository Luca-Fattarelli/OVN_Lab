class Signal_information:

    def __init__(self, sig_power_value: float, path: list):
        self.__signal_power = sig_power_value
        self.__path = path
        self.__latency = 0
        self.__noise_power = 0
    # TODO: change getter and setters
    # Getters and setters

    def get_signal_power(self):
        return self.__signal_power

    def set_signal_power(self, value: float):
        self.__signal_power = value

    def get_noise_power(self):
        return self.__noise_power

    def set_noise_power(self, value: float):
        self.__noise_power = value

    def get_latency(self):
        return self.__latency

    def set_latency(self, value: float):
        self.__latency = value

    def get_path(self):
        return self.__path

    # TODO: check if str is good for char or if i have to make a more thorough check

    def set_path(self, path: list[str]):
        self.__path = path

    def sig_power_increase(self, increment: float):
        self.__signal_power += increment

    def noise_power_increase(self, increment: float):
        self.__noise_power += increment

    def latency_increase(self, increment: float):
        self.__latency += increment

    def path_update(self):
        return self.__path.pop(0)

