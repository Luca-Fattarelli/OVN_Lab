import lightpath
import signal_information
import line


class Node:
    def __init__(self, label: str, position, connected_nodes, transceiver='fixed-rate'):
        self.label = label
        self.position = position
        self.connected_nodes = connected_nodes
        self.transceiver = transceiver
        self.successive = {}
    # TODO: getter a

    def set_transceiver(self, transceiver):
        self.transceiver = transceiver

    def get_transceiver(self):
        return self.transceiver

    def get_label(self):
        return self.label

    def get_connected_nodes(self):
        return self.connected_nodes

    def get_position(self):
        return self.position

    def set_successive(self, label: str, line1):
        self.successive[label] = line1

    def propagate(self, sig: lightpath.Lightpath):
        next_label = sig.path_update()
        if sig.get_path():
            next_label += sig.get_path()[0]
        if next_label is None:
            return
        next_line = self.successive.get(next_label)
        channel = sig.get_channel()
        if next_line is None:
            return
        if next_line.get_state(channel) == "1" and next_line.get_in_service:
            next_line.set_opt_launch_power(next_line.optimized_launch_power())
            next_line.propagate(sig)
        else:
            return

    def probe(self, sig_info: lightpath.Lightpath):
        next_label = sig_info.path_update()
        if sig_info.get_path():
            next_label += sig_info.get_path()[0]
        if next_label is None:
            return
        next_line = self.successive.get(next_label)
        if next_line is None:
            return
        if (type(sig_info).__name__ is signal_information.Signal_information.__name__
                or next_line.get_state(sig_info.get_channel()) == "1") and next_line.get_in_service():
            next_line.probe(sig_info)
        else:
            return
