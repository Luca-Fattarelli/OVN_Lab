import lightpath
import signal_information
import line


class Node:
    def __init__(self, label: str, position, connected_nodes):
        self.label = label
        self.position = position
        self.connected_nodes = connected_nodes
        self.successive = {}
    # TODO: getter a

    def get_label(self):
        return self.label

    def get_connected_nodes(self):
        return self.connected_nodes

    def get_position(self):
        return self.position

    def set_successive(self,label: str, line1):
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
        if next_line.get_state(channel) == "1":
            next_line.propagate(sig)
        else:
            print("EEEEEEh?", channel, sig.get_path())
            return

    def probe(self, sig_info: lightpath.Lightpath):
        next_label = sig_info.path_update()
        if sig_info.get_path():
            next_label += sig_info.get_path()[0]
        if next_label is None:
            #print("End\n")
            return
        next_line = self.successive.get(next_label)
        #print(self.successive)
        if next_line is None:
            # print("End(NOLINE)\n")
            return
        if type(sig_info).__name__ is signal_information.Signal_information.__name__ \
                or next_line.get_state(sig_info.get_channel()) == "1":

            next_line.probe(sig_info)
        else:
            return
        #print("Propagate into line: " + next_label + "\n")
        # line propagate method





#list1 = ["a", "b", "c"]
#sig_inf = Signal_information(10.0, list1)
#node = Node('a', None, None)
#node.propagate(sig_inf, None)
