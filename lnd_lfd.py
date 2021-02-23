#!/usr/bin/env python3
# Pipe input from `lightning-cli listchannels`
# pip3 install PyMaxflow mpmath
from functools import reduce
import maxflow
from mpmath import *
import numpy as np
from pathlib import Path
from helpers import getConfig
from os.path import join, exists
import csv


class LowFeeDiversityFinder:
    def __init__(self,
                 graph_json,
                 root_node_id,
                 base_fee_threshold=10000,
                 permillion_fee_threshold=100,
                 min_channels=10,
                 min_capacity=15000000
                 ):
        self.config = getConfig()
        self.benefits_dir = self.config["benefits_dir"]
        self.graph_json = graph_json
        self.name = self.graph_json["timestamp"]
        self.root = 0
        self.root_id = root_node_id

        # Define the "low fee" threshold
        # These values are for the *total path*
        self.base_fee_threshold = base_fee_threshold
        self.permillion_fee_threshold = permillion_fee_threshold

        # Define the minimum channels and capacity requirements to consider a node for an outgoing channel
        self.min_channels = min_channels
        self.min_capacity = min_capacity  # NOT about total capacity of a channel path

        self.outgoing = dict()
        self.incoming = dict()  # TODO: are we sure we need both of these? Channels do not have direction

        self.nodes = set()
        self.node_to_id = dict()
        self.id_to_node = dict()
        self.chan_fees = {}
        self.chan_capacity = {}
        self.outgoing = {}
        self.incoming = {}
        self.new_peer_benefit = {}

        self.nodes.add(self.root)
        self.node_to_id[self.root] = self.root_id
        self.id_to_node[self.root_id] = self.root

        self.flow_graph = None
        if exists(join(self.benefits_dir, str(self.name) + "-graph.csv")):
            self.load_peer_benefits()
        else:
            self.find_lfd_peers()
            self.save_peer_benefits()

    def find_lfd_peers(self):

        i = 1
        num_inactive_channels = 0
        for chan in self.graph_json["graph"]["edges"]:
            src_id = chan["node1_pub"]
            if src_id not in self.id_to_node:
                self.node_to_id[i] = src_id
                self.id_to_node[src_id] = i
                self.nodes.add(i)
                i += 1
            src = self.id_to_node[src_id]

            dest_id = chan["node2_pub"]
            if dest_id not in self.id_to_node:
                self.node_to_id[i] = dest_id
                self.id_to_node[dest_id] = i
                self.nodes.add(i)
                i += 1
            dest = self.id_to_node[dest_id]

            # NOTE: node1_policy = node2_policy, you can use either
            if chan["node1_policy"] is None or chan["node1_policy"]["disabled"]:
                num_inactive_channels += 1
            else:
                if src not in self.outgoing:
                    self.outgoing[src] = set()
                self.outgoing[src].add(dest)
                if dest not in self.incoming:
                    self.incoming[dest] = set()
                self.incoming[dest].add(src)

                # NOTE: node1_policy = node2_policy, you can use either
                base_fee = int(chan["node1_policy"]["fee_base_msat"])
                permillion_fee = int(chan["node1_policy"]["fee_rate_milli_msat"])
                self.chan_fees[(src, dest)] = (permillion_fee, base_fee)
                self.chan_capacity[(src, dest)] = int(chan["capacity"])

        print("Finished first stage.")

        num_active_nodes = sum(map(lambda n: 1 if n in self.outgoing or n in self.incoming else 0, self.nodes))
        print("{}/{} active/total nodes and {}/{} active/total (unidirectional) channels found.".format(
            num_active_nodes, len(self.nodes), len(self.chan_fees) - num_inactive_channels, len(self.chan_fees)))
        self.nodes.remove(self.root)

        existing_reachable_nodes = self.get_lowfee_reachable_node_maxflows()
        maxflow_sum = sum([n[1] for n in existing_reachable_nodes.items()])
        maxflow_prod = np.prod([mpf(n[1]) for n in existing_reachable_nodes.items()])
        if maxflow_sum == 0:
            maxflow_mean = 0
        else:
            maxflow_mean = float(maxflow_sum) / float(len(existing_reachable_nodes))
        if len(existing_reachable_nodes) == 0:
            maxflow_geomean = 0
        else:
            maxflow_geomean = power(maxflow_prod, mpf(1.0) / mpf(len(existing_reachable_nodes)))
        print("{} \"low-fee reachable\" nodes already exist with mean and geomean route diversity {} and {}.".format(
            len(existing_reachable_nodes), maxflow_mean, nstr(maxflow_geomean, 6)))

        # Iterate over all other nodes, sorted by decreasing number of incoming channels under the theory that more connected nodes
        # are more likely to have higher peer benefit, thus giving good answers more quickly
        i = 0
        nodes_num_outgoing = {n: len(self.outgoing[n]) if n in self.outgoing else 0 for n in self.nodes}
        for n in [k for k, v in sorted(nodes_num_outgoing.items(), key=lambda x: x[1], reverse=True)]:
            if n in self.outgoing.get(self.root, []):
                continue
            if not self.node_is_big_enough(n):
                continue
            if i % 100 == 0:
                if i == 0:
                    print("Trying new peers.")
                else:
                    print("Tried %d peers:\n----------" % i)
                    self.print_top_new_peers(10)
                    print("----------")
            now_reachable = self.get_lowfee_reachable_node_maxflows(n)
            maxflow_prod = mpf('1.0')
            num_new_nodes = 0
            routability_improvements = 0
            bonus = 0
            for r in now_reachable:
                maxflow_prod *= now_reachable[r]
                if r not in existing_reachable_nodes:
                    num_new_nodes += 1
                elif now_reachable[r] > existing_reachable_nodes[r]:
                    routability_improvements += 1
                    if existing_reachable_nodes[r] < 3:
                        bonus += 3 - existing_reachable_nodes[r]
            self.new_peer_benefit[n] = 3 * num_new_nodes + routability_improvements + bonus
            maxflow_geomean = power(maxflow_prod, mpf('1.0') / mpf(len(now_reachable)))
            print(
                "Peer %s has benefit %f with %d new low-fee reachable nodes and %d low-fee routability improvements, bonus %d; would make maxflow geomean %s" % (
                    self.node_to_id[n], self.new_peer_benefit[n], num_new_nodes, routability_improvements, bonus,
                    nstr(maxflow_geomean, 6)))
            i += 1

        self.print_top_new_peers(10)

    def node_is_big_enough(self, n):
        num_channels = 0
        total_capacity = 0
        if n in self.outgoing:
            num_channels += len(self.outgoing[n])
            total_capacity += reduce(lambda x, y: x + y, [self.chan_capacity[(n, o)] for o in self.outgoing[n]])
        if n in self.incoming:
            num_channels += len(self.incoming[n])
            total_capacity += reduce(lambda x, y: x + y, [self.chan_capacity[(i, n)] for i in self.incoming[n]])

        return num_channels >= self.min_channels and total_capacity >= self.min_capacity

    def print_top_new_peers(self, num):
        count = 0
        for (n, b) in sorted(self.new_peer_benefit.items(), key=lambda x: x[1], reverse=True):
            if n in self.incoming.get(self.root, []) or n in self.outgoing.get(self.root, []):
                continue
            elif not self.node_is_big_enough(n):
                continue
            else:
                print(f"{b} benefit from peering with node {self.node_to_id[n]}")
                count += 1
                if count >= num:
                    break

    def get_unweighted_maxflow(self, sink, edges, node_map, source_cap, count):

        self.flow_graph.reset()
        self.flow_graph.add_nodes(count)

        sink_cap = 0
        for (src, dest) in edges:
            g_src = node_map[src]
            g_dest = node_map[dest]
            self.flow_graph.add_edge(g_src, g_dest, 1, 0)

            if dest == sink:
                sink_cap += 1

        self.flow_graph.add_tedge(node_map[self.root], source_cap, 0)
        self.flow_graph.add_tedge(node_map[sink], 0, sink_cap)

        return self.flow_graph.maxflow()

    def get_lowfee_reachable_node_maxflows(self, proposed_new_peer=None, max_hops=None):
        lowfee_maxflows = dict()
        lowfee_reachable = set()
        lowfee_edges = set()
        min_cost_to_node = dict()  # maps to a list of 2 fee tuples (permillion, base) for minimum permillion fee and minimum base fee
        processed_nodes = set()
        queued = set()
        if max_hops == None:
            max_hops = 20

        min_cost_to_node[self.root] = [(0, 0), (0, 0)]  # [feerate_min_permillion, feerate_min_base]
        processed_nodes.add(self.root)
        queued.add(self.root)
        bfs_queue = [(n, 1) for n in self.outgoing.get(self.root, [])]
        for o in self.outgoing.get(self.root, []):
            min_cost_to_node[o] = [(0, 0), (0, 0)]
            lowfee_edges.add((self.root, o))
            queued.add(o)

        if proposed_new_peer is not None:
            lowfee_edges.add((self.root, proposed_new_peer))
            min_cost_to_node[proposed_new_peer] = [(0, 0), (0, 0)]
            queued.add(proposed_new_peer)
            bfs_queue.append((proposed_new_peer, 1))
        # use (0, 0) here instead of chan_fees[(root_node, n)] because we control these fees and they're independent of the peer node's low-fee reachability

        while len(bfs_queue) > 0:
            (cur_node, cur_hops) = bfs_queue.pop(0)
            processed_nodes.add(cur_node)
            min_feerates = min_cost_to_node[cur_node]
            for min_feerate in min_feerates:
                (permillion_fee, base_fee) = min_feerate

                if permillion_fee <= self.permillion_fee_threshold and base_fee <= self.base_fee_threshold:
                    lowfee_reachable.add(cur_node)
                else:
                    continue

                if cur_node not in self.outgoing:
                    continue

                for o in self.outgoing[cur_node]:
                    (new_permillion_fee, new_base_fee) = self.chan_fees[(cur_node, o)]
                    new_permillion_fee += permillion_fee
                    new_base_fee += base_fee

                    t1 = False
                    if new_permillion_fee <= self.permillion_fee_threshold and new_base_fee <= self.base_fee_threshold:
                        t1 = True
                        lowfee_edges.add((cur_node, o))

                    if o not in min_cost_to_node:
                        min_cost_to_node[o] = [(new_permillion_fee, new_base_fee), (new_permillion_fee, new_base_fee)]
                    if new_permillion_fee < min_cost_to_node[o][0][0]:
                        min_cost_to_node[o][0] = (new_permillion_fee, new_base_fee)
                    if new_base_fee < min_cost_to_node[o][1][1]:
                        min_cost_to_node[o][1] = (new_permillion_fee, new_base_fee)

                    t2 = o not in processed_nodes
                    t3 = cur_hops < max_hops
                    if t1 and t2 and t3:
                        if o not in queued:
                            queued.add(o)
                            bfs_queue.append((o, cur_hops + 1))

        source_cap, node_map, count = self.preprocess(lowfee_edges)
        for cur_node in lowfee_reachable:
            # calculate the maxflow from root_node -> cur_node with all channels having unit weight
            lowfee_maxflows[cur_node] = self.get_unweighted_maxflow(cur_node, lowfee_edges, node_map, source_cap, count)
        return lowfee_maxflows

    def preprocess(self, lowfee_edges):
        node_map = dict()
        source_cap = 0
        i = 0
        for (src, dest) in lowfee_edges:
            if src not in node_map:
                node_map[src] = i
                i += 1
            if dest not in node_map:
                node_map[dest] = i
                i += 1
            if src == self.root:
                source_cap += 1

        self.flow_graph = maxflow.Graph[int](i, len(lowfee_edges))

        return source_cap, node_map, i

    def save_peer_benefits(self):
        Path(self.benefits_dir).mkdir(exist_ok=True)
        with open(join(self.benefits_dir, str(self.name) + "-graph.csv"), "w") as csvfile:
            csv_writer = csv.writer(csvfile, lineterminator='\r')
            csv_writer.writerow(["Node", "Benefit", "Connected", "Big_Enough"])
            for (node, benefit) in sorted(self.new_peer_benefit.items(), key=lambda x: x[1], reverse=True):
                connected = node in self.incoming.get(self.root, []) or node in self.outgoing.get(self.root, [])
                node_id = self.node_to_id[node]
                big_enough = self.node_is_big_enough(node)
                csv_writer.writerow(map(str, [node_id, benefit, connected, big_enough]))

    def load_peer_benefits(self):
        self.new_peer_benefit = []
        with open(join(self.benefits_dir, str(self.name) + "-graph.csv"), "r") as csvfile:
            csv_reader = csv.reader(csvfile, lineterminator='\r')
            headers = next(csv_reader)

            for row in csv_reader:
                self.new_peer_benefit.append(dict(zip(headers, row)))
