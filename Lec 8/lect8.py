import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from math import radians, sin, cos, sqrt, atan2

# Code mostly from previous lecture

########################################
# Representing graphs
########################################

class Digraph:
    """Represents a weighted directed graph as adjacency list,
       mapping each node to outgoing edges and their weights.
       Nodes are strings"""

    def __init__(self, nodes=()):
        self._edges = {}
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        if node in self._edges:
            raise ValueError('Duplicate node')
        self._edges[node] = {}

    def add_edge(self, src, dest, weight = 1):
        if src not in self._edges:
            self.add_node(src)
        if dest not in self._edges:
            self.add_node(dest)
        self._edges[src][dest] = weight

    def has_node(self, node):
        return node in self._edges

    def get_all_nodes(self):
        return list(self._edges.keys())

    def outgoing_edges_of(self, node):
        return self._edges[node].copy()

    def children_of(self, node):
        return list(self.outgoing_edges_of(node).keys())

    # cost method added in lecture 8
    def cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            try:
                cost += self._edges[path[i]][path[i+1]]
            except:
                raise ValueError('Bad path')
        return cost

    def __str__(self):
        vals = []
        for src in self._edges:
            entry = src + ': '
            for dest, weight in self._edges[src].items():
                entry += f'{dest}({weight}), '
            if entry[-2:] != ': ': # There was at least one edge
                vals.append(entry[:-2])
            else:
                vals.append(entry[:-1])
        vals.sort(key=lambda x: x.split(':')[0])
        result = ''
        for v in vals:
            result += v + '\n'
        return result[:-1]


class Graph(Digraph):
    """Represents an undirected graph with pairs of directed edges"""

    def add_edge(self, node1, node2, weight=1):
        super().add_edge(node1, node2, weight)
        super().add_edge(node2, node1, weight)


# ########################################
# # Example graphs
# ########################################

# def build_city_graph():
#     g = Digraph()
#     g.add_edge('Boston', 'Providence')
#     g.add_edge('Boston', 'New York')
#     g.add_edge('Providence', 'Boston')
#     g.add_edge('Providence', 'New York')
#     g.add_edge('New York', 'Chicago')
#     g.add_edge('Chicago', 'Denver')
#     g.add_edge('Chicago', 'Phoenix')
#     g.add_edge('Denver', 'Phoenix')
#     g.add_edge('Denver', 'New York')
#     g.add_edge('Los Angeles', 'Boston')
#     return g

# def build_test_graph(graph_type):
#     g = graph_type()
#     g.add_edge('A', 'B')
#     g.add_edge('A', 'C')
#     g.add_edge('A', 'D')
#     g.add_edge('B', 'E')
#     g.add_edge('C', 'E')
#     g.add_edge('C', 'F')
#     g.add_edge('C', 'G')
#     g.add_edge('E', 'H')
#     g.add_edge('F', 'H')
#     g.add_edge('G', 'H')
#     return g

# # g = build_test_graph(Digraph)
# # print('An example directed graph:')
# # print(g)
# # print()

# # g = build_test_graph(Graph)
# # print('An example undirected graph:')
# # print(g)
# # print()

def path_to_string(path):
    """path is a list of nodes"""
    if path == None:
        return 'None'
    result = ''
    for node in path:
        result += node + '->'
    return result[:-2]

def pathlist_to_string(queue):
    """queue is a list of nodes"""
    result = ''
    for path in queue:
        result += path_to_string(path) + ', '
    return '[' + result[:-2] + ']'

def bfs_graph(graph, source, target, verbose=False):
    current_frontier = [[source]]
    next_frontier = []
    visited = set()
    visited.add(source)

    while current_frontier:
        if verbose:
            print('Current frontier:', pathlist_to_string(current_frontier))

        for path in current_frontier:
            if verbose:
                print('  Current BFS path:', path_to_string(path))

            last_node = path[-1]
            for next_node in graph.children_of(last_node):
                if next_node in visited:
                    continue
                new_path = path + [next_node]
                visited.add(next_node)
                if next_node == target:
                    if verbose:
                        print('Path found')
                    return new_path
                next_frontier.append(new_path)

        current_frontier, next_frontier = next_frontier, []
    return None


########### Code new to this lecture ###############

########################################
# Generalizing to find SHORTEST paths in graphs
########################################


def dfs_graph(graph, source, target, verbose=False):

    def dfs_internal(path, shortest):
        if verbose:
            print('Current DFS path:', path_to_string(path))

        last_node = path[-1]
        if last_node == target:
            if verbose:
                print(f'Path of length {len(path)} found')
            return path

        if shortest and len(path) + 1 >= len(shortest):
            return None

        best_path = None
        for next_node in graph.children_of(last_node):
            if next_node in path:
                continue
            new_path = dfs_internal(path + [next_node], shortest)
            if new_path:
                if not best_path or len(new_path) < len(best_path):
                    best_path = new_path
                if not shortest or len(new_path) < len(shortest):
                    shortest = new_path
        return best_path

    return dfs_internal([source], None)

# g = build_test_graph(Digraph)
# print(dfs_graph(g, 'A', 'H', verbose=True))
# print()

# g = build_city_graph()
# # g.add_edge('Providence', 'Phoenix')
# print(dfs_graph(g, 'Boston', 'Phoenix', verbose=True))
# sys.exit()

def dfs_weighted(g, source, target, verbose=False):

    def dfs_internal(path, shortest):
        if verbose:
            print(f'Current DFS path of cost {g.cost(path)}: '
                  f'{path_to_string(path)}')
        last_node = path[-1]
        if last_node == target:
            if verbose:
                print(f'Path of cost {g.cost(path)} found')
            return path
        if shortest and g.cost(path) >= g.cost(shortest):
            return None
        best_path = None
        for next_node in g.children_of(last_node):
            if next_node in path:
                continue
            new_path = dfs_internal(path + [next_node], shortest)
            if new_path:
                if not best_path or g.cost(new_path) < g.cost(best_path):
                    best_path = new_path
                if not shortest or g.cost(new_path) < g.cost(shortest):
                    shortest = new_path
        return best_path

    return dfs_internal([source], None)


# # weighted vs unweighted paths
# g = Digraph()
# g.add_edge('a', 'b', 1)
# g.add_edge('b', 'c', 1)
# g.add_edge('c', 'd', 2)
# g.add_edge('a', 'c', 5)
# g.add_edge('d', 'c', 1)
# print('A weighted digraph')
# print(g, '\n')

# print('Try unweighted DFS')
# print(f"Path found by dfs_graph: {dfs_graph(g, 'a', 'd')}\n")
# print('Try weighted DFS')
# print(f"Path found by dfs_weighted: {dfs_weighted(g, 'a', 'd')}")
# sys.exit()

# Testing code

def build_random_graph(graph_type, num_nodes, num_edges):
    """Generates a random graph of type graph_type with
    num_nodes nodes and num_edges edges."""
    if num_nodes <= 0 or num_edges < 0:
        raise ValueError("Number of nodes must be positive,\
                         and number of edges cannot be negative.")
    # Create an empty graph of the specified type
    graph = graph_type()
    # Generate random graph
    nodes = [str(i) for i in range(num_nodes)]
    for _ in range(num_edges):
        src = random.choice(nodes)
        dest = random.choice(nodes)
        weight = random.randint(1, 10)
        graph.add_edge(src, dest, weight)
    return graph


def Dijkstra(graph, start, end, to_print = False):
    """
    graph: a weighted  digraph
        all the weights are non-negative
    start: a node in  graph
    end: a node in graph
    returns a list representing shortest path from start to end,
       and None if no path exists"""

    # Mark all nodes unvisited and store them.
    # Set the distance to zero for our initial node
    # and to infinity for other nodes.
    unvisited = graph.get_all_nodes()
    distance_to = {node: float('inf') for node in graph.get_all_nodes()}
    distance_to[start] = 0
    # Mark all nodes as not having found a predecessor node on pathfrom start
    predecessor = {node: None for node in graph.get_all_nodes()}

    while unvisited:
        # Select the unvisited node with the smallest distance from
        # start, it's current node now.
        current = min(unvisited, key=lambda node: distance_to[node])
        if to_print: #for pedagocical purposes
            print('\nValue of current:', current)
            print('Value of distance_to:')
            for k in distance_to:
                print('  ' + k + ':', distance_to[k])
            print('Value of predecessor:')
            for k in predecessor:
                print('  ' + k + ':', predecessor[k])

        # Stop, if the smallest distance
        # among the unvisited nodes is infinity.
        if distance_to[current] == float('inf'):
            break

        # Find unvisited neighbors for the current node
        # and calculate their distances from start through the
        # current node.
        for neighbour in graph.children_of(current):
            dist = graph.outgoing_edges_of(current)[neighbour]
            alternative_path_dist = distance_to[current] + dist #hops as distance

            # Compare the newly calculated distance to the assigned.
            # Save the smaller distance and update predecssor.
            if alternative_path_dist < distance_to[neighbour]:
                distance_to[neighbour] = alternative_path_dist
                predecessor[neighbour] = current

        # Remove the current node from the unvisited set.
        unvisited.remove(current)

    #Attempt to be build a path working backwards from end
    path = []
    current = end
    while predecessor[current] != None:
        path.insert(0, current)
        current = predecessor[current]
    if path != []:
        path.insert(0, current)
    else:
        return None
    return path


def build_city_graph():
    g = Digraph()
    g.add_edge('Boston', 'Providence')
    g.add_edge('Boston', 'New York')
    g.add_edge('Providence', 'Boston')
    g.add_edge('Providence', 'New York')
    g.add_edge('New York', 'Chicago')
    g.add_edge('Chicago', 'Denver')
    g.add_edge('Chicago', 'Phoenix')
    g.add_edge('Denver', 'Phoenix')
    g.add_edge('Denver', 'New York')
    g.add_edge('Los Angeles', 'Boston')
    return g

# g = build_city_graph()
# path = Dijkstra(g, 'Boston', 'Chicago', to_print = True)
# print(path_to_string(path))
# sys.exit()

random.seed(2)
dij_times, bfs_times, x_vals = [], [], []

def one_trial(destination = 'unreachable'):
    for num_nodes in range(100, 6001, 100):
        x_vals.append(num_nodes)
        g = build_random_graph(Digraph, num_nodes, 5*num_nodes)
        g.add_node('unreachable')
        print(f'For digraph with V = {num_nodes} and E = {5*num_nodes}')
        start = time.time()
        bfs_graph(g, '0', destination, verbose = False)
        time_taken = time.time() - start
        bfs_times.append(time_taken)
        print(f'Seconds for bfs: {time_taken:.6f}')
        start = time.time()
        Dijkstra(g, '0', destination)
        time_taken = time.time() - start
        dij_times.append(time_taken)
        print(f'Seconds for Dijkstra: {time_taken:.6f}')
    return(bfs_times, dij_times)

destination = '98'
bfs_times, dij_times = one_trial(destination)

plt.plot(x_vals, dij_times, label = 'Dijkstra')
plt.plot(x_vals, bfs_times, label = 'BFS')
plt.title(f'Time Looking for Path from 0 to {destination}\n(|E| = 5*|V|)')
plt.xlabel('Number of Nodes')
plt.ylabel('Time in Seconds')
plt.legend()
plt.semilogy()
# sys.exit()


# distance metrics for A*

# Compute distances on a spherical approximation of Earth
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000  # Earth's radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c  # distance in meters

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def Euclidean_dist(n1, n2, dict_coord):
    p1=dict_coord[n1]
    p2=dict_coord[n2]
    return  distance(p1, p2)

def dist_great_circle(n1, n2, dict_coord):
    p1=dict_coord[n1]
    p2=dict_coord[n2]
    return haversine_distance(p1[1], p1[0], p2[1], p2[0])/1000

def AStar(graph, start, end, heuristic, dict_coord, to_print = False):
    """
    graph: a weighted digraph
    start: a node in  graph
    end: a node in graph
    returns a list representing shortest path from start to end,
       and None if no path exists"""

    # Mark all nodes unvisited and store them.
    # Set the distance to zero for our initial node
    # and to infinity for other nodes.
    #unvisited = graph.get_all_nodes()
    distance_to = {node: float('inf') for node in graph.get_all_nodes()}
    distance_to[start] = 0
    heuristic_to = {node: heuristic(node, end, dict_coord) for node in graph.get_all_nodes()}

    # Mark all nodes as not having found a predecessor node on pathfrom start
    predecessor = {node: None for node in graph.get_all_nodes()}
    open_set = [start]
    while open_set:
        # Select the unvisited node with the smallest distance from
        # start, it's current node now.
        current = min(open_set, key=lambda node: heuristic_to[node])

        if to_print: #for pedagogical purposes
            print('\nValue of current:', current)
            print('Value of distance_to:')
            for k in distance_to:
                print('  ' + k + ':', distance_to[k])
            print('Value of predecessor:')
            for k in predecessor:
                print('  ' + k + ':', predecessor[k])

        # Stop, if the smallest distance
        # among the unvisited nodes is infinity.
        if distance_to[current] == float('inf'):
            break
        if current == end:
            break
        # Find unvisited neighbors for the current node
        # and calculate their distances from start through the
        # current node.
        for neighbour in graph.children_of(current):
            dist = graph.outgoing_edges_of(current)[neighbour]
            alternative_path_dist = distance_to[current] + dist
            # Compare the newly calculated distance to the assigned.
            # Save the smaller distance and update predecssor.
            if alternative_path_dist < distance_to[neighbour]:
                distance_to[neighbour] = alternative_path_dist
                predecessor[neighbour] = current
                #update heuristic
                heuristic_to[neighbour]=alternative_path_dist + heuristic(neighbour, end, dict_coord)
                if not neighbour in open_set:
                    open_set.append(neighbour)

        # Remove the current node from the unvisited set.
        open_set.remove(current)

    #Attempt to be build a path working backwards from end
    path = []
    current = end
    while predecessor[current] != None:
        path.insert(0, current)
        current = predecessor[current]
    if path != []:
        path.insert(0, current)
    else:
        return None
    return path
