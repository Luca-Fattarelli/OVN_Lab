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

    def propagate(self, sig_info: signal_information.Signal_information):
        next_label = sig_info.path_update()
        if sig_info.get_path():
            next_label += sig_info.get_path()[0]
        if next_label is None:
            print("End\n")
            return
        next_line = self.successive.get(next_label)
        #print(self.successive)
        if next_line is None:
            #print("End(NOLINE)\n")
            return
        #print("Propagate into line: " + next_label + "\n")
        # line propagate method
        next_line.propagate(sig_info)




#list1 = ["a", "b", "c"]
#sig_inf = Signal_information(10.0, list1)
#node = Node('a', None, None)
#node.propagate(sig_inf, None)