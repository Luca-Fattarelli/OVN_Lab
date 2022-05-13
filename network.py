import numpy
import random
import connection
import lightpath
import line
import node
import signal_information
import json
import math
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def min_distance(x1: float, x2, y1, y2):
    a = (y2 - y1)
    b = (x2 - x1)
    res = a * a + b * b
    return math.sqrt(res)


def calculate_snr(sig):
    #print(sig.get_noise_power())
    x = sig.get_signal_power() / sig.get_noise_power()
    r = 10 * numpy.log10(x)
    return r


class Network:
    channel_num = 10


    def __init__(self):
        self.nodes = {}
        self.lines = {}
        f = open('nodes.json')
        data = {}
        data = json.load(f)

        for v in data:
            single_data = data.get(v)
            lines_to_be_made = single_data.get("connected_nodes")
            node1 = node.Node(v, single_data.get("position"), lines_to_be_made)
            self.nodes[v] = node1

        for a in self.nodes:
            n = self.nodes.get(a)
            for b in n.get_connected_nodes():
                second = self.nodes.get(b)
                line_name = a + b
                if line_name not in self.lines:
                    # add line
                    c = n.get_position()
                    d = second.get_position()
                    length = min_distance(c[0], d[0], c[1], d[1])
                    line1 = line.Line(line_name, length)
                    #to set up channel as free I use the variable channel_num a
                    line1.setup_state(self.channel_num)
                    #line1.set_successive(b, second)
                    self.lines[line_name] = line1
                    #self.nodes[a].set_successive(line_name, line1)
                    #mod
                    #line_name = b + a
                    #line2 = line.Line(line_name, length)
                    #line2.setup_state(self.channel_num)
                    #self.lines[line_name] = line2

    def connect(self):
        for a in self.nodes:
            n = self.nodes.get(a)
            for b in n.get_connected_nodes():
                second = self.nodes.get(b)
                line1 = self.lines.get(a+b)
                line1.set_successive(b, second)
                self.nodes[a].set_successive(a+b, line1)

    def find_paths(self,label1 :str, label2: str):
        start_node = self.nodes.get(label1)
        end_node = self.nodes.get(label2)
        self.res = []
        path = [label1]
        self.pathR(label2, path, start_node)
        res = self.res
        del self.res
        return res

    def pathR(self, end_label: str, path: list[str], node1: node.Node):
        next1 = node1.get_connected_nodes()

        for n1 in next1:
            if n1 in path:
                continue
            path.append(n1)
            if n1 is end_label:
                path2 = path.copy()
                self.res.append(path2)
            else:
                #copio lista
                self.pathR(end_label, path, self.nodes.get(n1))
            path.pop(-1)

    def propagate(self, sig):
        start_label = sig.get_path()[0]
        start_node = self.nodes.get(start_label)
        start_node.propagate(sig)

    def probe(self, sig_info):
        start_label = sig_info.get_path()[0]
        start_node = self.nodes.get(start_label)
        start_node.probe(sig_info)

    def draw(self):
        plt.grid()

        for i in self.nodes:
            node1 = self.nodes.get(i)
            pos = node1.get_position()
            plt.plot(pos[0], pos[1], marker="o", markersize=20)
            plt.annotate(i, pos, textcoords="offset points", xytext=(0, 10))
        for l in self.lines:
            n1 = self.nodes.get(l[0])
            n2 = self.nodes.get(l[1])
            pos1 = n1.get_position()
            pos2 = n2.get_position()
            x_values = [pos1[0], pos2[0]]
            y_values = [pos1[1], pos2[1]]
            plt.plot(x_values, y_values, linestyle="--")
        plt.show()

    def create_route_space(self):
        names = []
        channels = []
        #columns for each channel
        for i in range(0, self.channel_num):
            channels.append([])
        #find all paths
        for a in self.nodes:
            for b in self.nodes:
                if a is not b:
                    paths = self.find_paths(a,b)
                    for path1 in paths:
                        tmp = []
                        for p in path1:
                            tmp.append(p)
                            tmp.append("->")
                        tmp.pop(-1)
                        res = "".join(tmp)
                        #create a lightpath for each channel to check availabilty
                        for i in range(0, self.channel_num):
                            path_copy = path1.copy()
                            lightp = lightpath.Lightpath(0.001, path_copy, i)
                            self.probe(lightp)
                            # if the path after the propagate method is not empty -> path is occupied
                            # empty list is considered false in python: full (true) means not empty
                            if lightp.get_path():
                                channels[i].append("0")
                            else:
                                channels[i].append("1")
                        names.append(res)
        #print(names)
        dataD = {}
        for i in range(0, self.channel_num):
            string = "Channel " + str(i)
            dataD[string] = channels[i]
        self.route_space =pd.DataFrame(dataD, names)


    def create_data_frame(self):
        names = []
        latency = []
        noise = []
        ratio = []
        for a in self.nodes:
            for b in self.nodes:
                if a is not b:
                    paths = self.find_paths(a, b)
                    for path1 in paths:
                        tmp = []
                        for p in path1:
                            tmp.append(p)
                            tmp.append("->")
                        tmp.pop(-1)
                        res = "".join(tmp)
                        sig = signal_information.Signal_information(0.001, path1)
                        self.probe(sig)
                        names.append(res)
                        noise.append(sig.get_noise_power())
                        latency.append(sig.get_latency())
                        r = calculate_snr(sig)
                        ratio.append(r)
        dataD = {"Latency": latency, "Noise": noise, "Signal to noise ratio": ratio}
        dataframe = pd.DataFrame(dataD, names)
        #display(dataframe)
        self.weighted_paths = dataframe

    def is_path_free(self, path: list[str]):
        # Used before route_space
        path_l = path.split("->")
        for i in range(0, len(path_l) - 1):
            label = path_l[i] + path_l[i + 1]
            #print(label)
            if self.lines.get(label).get_state() == 0:
                return False
        return True

    def occupy_path(self, path: list[str], channel: int):
        self.create_route_space()

    def free_all(self):
        for line1 in self.lines.values():
            line1.setup_state(self.channel_num)

    def find_best_snr(self, a: str, b: str):
        """OLD VERSION(no route_space)
        best_path = None
        best_ratio = None
        for index, row in self.weighted_paths.iterrows():
            # print(row["Signal to noise ratio"])
            if index[0] is a and index[-1] is b:
                if best_ratio is None or row["Signal to noise ratio"] > best_ratio:
                    if self.is_path_free(index):
                        best_ratio = row["Signal to noise ratio"]
                        best_path = index
        return best_path"""
        best_path = None
        best_snr = None
        channel = -1
        for index, row in self.weighted_paths.iterrows():
            if index[0] is a and index[-1] is b:
                if best_snr is None or row["Signal to noise ratio"] > best_snr:
                    r = self.route_space.loc[[index]]
                    for i in range(0, self.channel_num):
                        string = "Channel " + str(i)
                        val = r[string]
                        if val.item() == "1":
                            best_snr = row["Latency"]
                            best_path = index
                            channel = i
                            break
        return best_path, channel

    def find_best_latency(self, a: str, b: str):
        #finds avaialable path with best latency: returns path (and first available channel)
        best_path = None
        best_lat = None
        channel = -1
        for index, row in self.weighted_paths.iterrows():
            if index[0] is a and index[-1] is b:
                if best_lat is None or row["Latency"] < best_lat:
                    r = self.route_space.loc[[index]]
                    for i in range(0, self.channel_num):
                        string = "Channel " + str(i)
                        val = r[string]
                        if val.item() == "1":
                            best_lat = row["Latency"]
                            best_path = index
                            channel = i
                            break
        return best_path, channel


    def stream(self, connection_list: list[connection.Connection], label="Latency"):
        for elem in connection_list:
            if label == "SNR":
                path, channel = self.find_best_snr(elem.get_input(), elem.get_output())
                if path is None:
                    elem.set_latency(None)
                    elem.set_snr(0)
                    continue
                path_list = path.split("->")
                lp = lightpath.Lightpath(elem.get_signal_power(), path_list, channel)
            elif label == "Latency":
                path, channel = self.find_best_latency(elem.get_input(), elem.get_output())
                #print(path)
                if path is None:
                    elem.set_latency(None)
                    elem.set_snr(0)
                    continue
                path_list = path.split("->")
                #print(path_list)
                lp = lightpath.Lightpath(elem.get_signal_power(), path_list, channel)
            #print(lp.get_path())
            #elem.set_snr(self.weighted_paths.loc[[path]]["Signal to noise ratio"])
            #elem.set_snr(1)
            #print("Propagation", path, channel)
            self.propagate(lp)
            self.occupy_path(path, channel)
            elem.set_latency(lp.get_latency())
            elem.set_snr(calculate_snr(lp))

        #self.free_all()

def main():
    #es LAB 4
    network = Network()
    network.connect()
    network.create_data_frame()
    network.create_route_space()
    snr_collection = []
    lat_collection = []
    for i in range(0, 100):
        output = None
        input1 = random.choice(list(network.nodes.keys()))
        while output is None or output is input1:
            output = random.choice(list(network.nodes.keys()))
        con = connection.Connection(input1, output, 0.001)
        snr_collection.append(con)

    for i in range(0, 100):
        output = None
        input1 = random.choice(list(network.nodes.keys()))
        while output is None or output is input1:
            output = random.choice(list(network.nodes.keys()))
        con = connection.Connection(input1, output, 0.001)
        lat_collection.append(con)
    network.stream(lat_collection)
    print(network.route_space)
    network.free_all()
    network.stream(snr_collection, "SNR")
    print(network.route_space)
    #plt.rcParams.update()
    snr_list = []
    lat_list = []
    for i in range(0, 100):
        snr = snr_collection[i].get_snr()
        if snr == 0:
            snr = numpy.NaN
        snr_list.append(snr)
        lat = lat_collection[i].get_lat()
        if lat is None:
            lat = numpy.NaN
        lat_list.append(lat)
    plt.hist(snr_list, bins=30)
    plt.xlabel("SNR")
    plt.ylabel("Number of connections")
    plt.show()
    plt.hist(lat_list, bins=30)
    plt.xlabel("Latency")
    plt.ylabel("Number of connections")
    plt.show()



    path = ["A", "B"]
    #if the path after the propagate method is not None -> path is occupied
    """sig = lightpath.Lightpath(0,path,0)
    network.propagate(sig)
    print(sig.get_path())
    network.create_route_space()
    sig1 = lightpath.Lightpath(0, ["A", "B"], 1)
    network.propagate(sig1)
    print(sig1.get_path())
    #network.create_route_space()
    #print((network.route_space.loc[["A->B"]]))
    #print(network.find_best_snr("A","B"))
"""



if __name__ == "__main__":
    main()
