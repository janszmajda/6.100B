import random
import time
import matplotlib.pyplot as plt
import sys

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


########################################
# Example graphs
########################################


def build_test_graph(graph_type):
    g = graph_type()
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('A', 'D')
    g.add_edge('B', 'E')
    g.add_edge('C', 'E')
    g.add_edge('C', 'F')
    g.add_edge('C', 'G')
    g.add_edge('E', 'H')
    g.add_edge('F', 'H')
    g.add_edge('G', 'H')
    return g

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
            
# g = build_test_graph(Digraph)
# print('An example directed graph:')
# print(g)
# print(g._edges) # shouldn't do this, but here to show what rep looks like
# print()
# sys.exit()

# g = build_test_graph(Graph)
# print('An example undirected graph:')
# print(g)
# print()

# g = build_city_graph()
# print('The city graph:')
# print(g)
# print()


########################################
# Finding paths in trees
########################################


def build_filesystem_tree():
    g = Digraph()
    g.add_edge('root', 'school')
    g.add_edge('root', 'personal')
    g.add_edge('root', 'downloads')
    g.add_edge('school', 'spring25')
    g.add_edge('school', 'fall25')
    g.add_edge('spring23', '6.100B')
    g.add_edge('spring23', '8.02')
    g.add_edge('personal', 'photos')
    g.add_edge('personal', 'bills')
    g.add_edge('downloads', 'lect1')
    g.add_edge('downloads', 'lect2')
    g.add_edge('downloads', 'lect3')
    g.add_edge('lect2', 'code.py')
    g.add_edge('lect2', 'slides.pdf')
    g.add_edge('lect2', 'data.txt')
    return g


def path_to_string(path):
    """path is a list of nodes"""
    if path == None:
        return 'None'
    result = ''
    for node in path:
        result += node + '->'
    return result[:-2]


def pathlist_to_string(queue):
    """path is a list of nodes"""
    result = ''
    for path in queue:
        result += path_to_string(path) + ', '
    return '[' + result[:-2] + ']'


def dfs_tree(tree, root, target, verbose=False):

    def dfs_internal(path):
        if verbose:
            print('Current DFS path:', path_to_string(path))

        last_node = path[-1]
        if last_node == target:
            if verbose:
                print('Path found')
            return path

        for next_node in tree.children_of(last_node):
            new_path = dfs_internal(path + [next_node])
            if new_path:
                return new_path
        return None

    return dfs_internal([root])


def bfs_tree(tree, root, target, verbose = False):
    current_frontier = [[root]]
    next_frontier = []

    while current_frontier:
        if verbose:
            print('Current frontier:', pathlist_to_string(current_frontier))

        for path in current_frontier:
            if verbose:
                print('  Current BFS path:', path_to_string(path))

            last_node = path[-1]
            for next_node in tree.children_of(last_node):
                new_path = path + [next_node]
                if next_node == target:
                    if verbose:
                        print('Path found')
                    return new_path
                next_frontier.append(new_path)

        current_frontier, next_frontier = next_frontier, []
    return None


g = build_filesystem_tree()
# print(g)
# print()
# sys.exit()


def print_result(source, dest, path):
    if path != None:
        print(f'Path from {source} to {dest} is {path}')
    else:
        print(f'There is no path from {source} to {dest}')

# print('Find path from root to data.txt')    
# path = dfs_tree(g, 'root', 'data.txt', verbose=True)
# print_result('root', 'data.txt', path)
# print()
# print('Find path from root to pict.jpg')
# path = dfs_tree(g, 'root', 'pict.jpg', verbose=True)
# print_result('root', 'pict.jpg', path)
# print()
# sys.exit()

# print('Find path from root to data.txt')    
# path = bfs_tree(g, 'root', 'data.txt', verbose=True)
# print_result('root', 'data.txt', path)
# print()
# print('Find path from root to pict.jpg')
# path = bfs_tree(g, 'root', 'pict.jpg', verbose=True)
# print_result('root', 'pict.jpg', path)
# sys.exit()


########################################
# Generalizing to find SHORTEST paths in graphs
########################################

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

# g = build_test_graph(Digraph)
# print(bfs_graph(g, 'A', 'H', verbose=True))
# print()

# g = build_city_graph()
## g.add_edge('Providence', 'Phoenix')
# print(bfs_graph(g, 'Boston', 'Phoenix', verbose=True))

# sys.exit()

# def dfs_graph(graph, source, target, verbose=False):

#     def dfs_internal(path, shortest):
#         if verbose:
#             print('Current DFS path:', path_to_string(path))

#         last_node = path[-1]
#         if last_node == target:
#             if verbose:
#                 print(f'Path of length {len(path)} found')
#             return path

#         if shortest and len(path) + 1 >= len(shortest):
#             return None

#         best_path = None
#         for next_node in graph.children_of(last_node):
#             if next_node in path:
#                 continue
#             new_path = dfs_internal(path + [next_node], shortest)
#             if new_path:
#                 if not best_path or len(new_path) < len(best_path):
#                     best_path = new_path
#                 if not shortest or len(new_path) < len(shortest):
#                     shortest = new_path
#         return best_path

#     return dfs_internal([source], None)

def dfs_graph_with_shortest(graph, source, target, verbose=False):

    def dfs_internal(path, shortest):
        if verbose:
            print('Current DFS path:', path_to_string(path))

        last_node = path[-1]
        if last_node == target:
            if verbose:
                print(f'Path found of length {len(path)}')
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
# print(dfs_graph_with_shortest(g, 'A', 'H', verbose=True))
# print()

# g = build_city_graph()
# # g.add_edge('Providence', 'Phoenix')
# print(dfs_graph_with_shortest(g, 'Boston', 'Phoenix', verbose=True))
# sys.exit()

def bfs_fifo(graph, source, target, verbose=False):
    queue = [[source]]
    visited = set()
    visited.add(source)
    nodes_visited = 1

    while queue:
        if verbose:
            print('Current queue:', pathlist_to_string(queue))

        path = queue.pop(0)
        if verbose:
            print('  Current BFS path:', path_to_string(path))

        last_node = path[-1]
        for next_node in graph.children_of(last_node):
            nodes_visited += 1
            if next_node in visited:
                continue
            new_path = path + [next_node]
            visited.add(next_node)
            if next_node == target:
                if verbose:
                    print(f'Number of nodes visited by bfs = {nodes_visited:,}')
                    print('Path found')
                return new_path
            queue.append(new_path)
    print(f'Number of nodes visited by bfs = {nodes_visited:,}')
    print('Path not found')
    return None

def dfs_lifo(graph, source, target, verbose=False):
    stack = [[source]]
    best_path = None
    nodes_visited = 1
    while stack:
        if verbose:
            print('Current stack:', pathlist_to_string(stack))
        path = stack.pop(-1)
        if verbose:
            print('  Current DFS path:', path_to_string(path))
        if best_path and len(path) + 1 >= len(best_path):
            continue
        last_node = path[-1]
        for next_node in graph.children_of(last_node):
            nodes_visited += 1
            if next_node in path:
                continue
            new_path = path + [next_node]
            if next_node == target:
                if verbose:
                    print('Path found', new_path)
                if not best_path or len(new_path) > len(best_path):
                    best_path = new_path
                continue
            stack.append(new_path)
    print(f'Number of nodes visited by dfs = {nodes_visited:,}')
    if best_path is None:
        print('Path not found')
    return best_path

def build_random_graph(graph_type, num_nodes, num_edges):
    """Generates a random graph of type graph_type with
    num_nodes nodes and num_edges edges."""
    if num_nodes <= 0 or num_edges < 0:
        raise ValueError("Number of nodes must be positive,\
                         and number of edges cannot be negative.")
    
    nodes = [str(i) for i in range(num_nodes)]
    graph = graph_type(nodes)
    # Generate random edges
    for _ in range(num_edges):
        src = random.choice(nodes)
        dest = random.choice(nodes)
        weight = random.randint(1, 10)  # Random weight
        graph.add_edge(src, dest, weight)
    return graph

dfs_times, bfs_times, x_vals = [], [], []
for num_nodes in range(10, 161, 50):
    x_vals.append(num_nodes)
    g = build_random_graph(Digraph, num_nodes, 2*num_nodes)
    g.add_node('unreachable')
    print(f'For digraph with V = {num_nodes} and E = {2*num_nodes}')
    start = time.time()
    bfs_fifo(g, '0', 'unreachable')
    time_taken = time.time() - start
    bfs_times.append(time_taken)
    print(f'Seconds for bfs: {time_taken:.6f}')
    start = time.time()
    dfs_lifo(g, '0', 'unreachable')
    time_taken = time.time() - start
    dfs_times.append(time_taken)
    print(f'Seconds for dfs: {time_taken:.6f}')

# plt.plot(x_vals, dfs_times, label = 'DFS')
# plt.plot(x_vals, bfs_times, label = 'BFS')
# plt.title('Time Looking for a Non-existant Path (E = 2*V)')
# plt.xlabel('Number of Nodes')
# plt.ylabel('Time in Seconds')
# plt.legend()
# plt.semilogy()
# sys.exit()
