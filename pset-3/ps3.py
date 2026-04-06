################################################################################
# 6.100B Fall 2025
# Problem Set 3 — Modeling Gerrymandering using Graphs (Lilliput)
# Name: Jan Szmajda
# Collaborators: PyTutor
# Time: 6hrs
################################################################################


################################################################################
# READ ME:
# - This file is your WORKING SKELETON. Fill what's necessary for each function.
# - Do NOT rename this file or change function signatures / docstrings.
# - You may not import additional libraries beyond the ones already here.
# - You should NOT modify the helper module (helper.py) that we provide.
################################################################################

from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import random
import helper

from helper import build_town_graph, plot_graph, plot_voronoi_map


################################################################################
# Part I — Model Lilliput as a Graph
################################################################################

def make_lilliput():
    """
    Builds a Graph representation of Lilliput by calling the build_town_graph()
    from helper.py.

    Returns:
        g (Graph): Graph instance of Lilliput
        points: (optional) locations for Voronoi plotting
        vor: (optional) Voronoi object for plotting

    """
    # Generates 32 point grid where each point is a Town object
    # Graph object created storing all towns and tracking adjacency
    return build_town_graph()

def allocate_swingendians_for_town(num_swingendians, swing_mean = -1, swing_sd = 1):
    """
    Allocate swing voters to Bigendians or Smallendians for a town
    
    Parameters:
        num_swingendians (int): Number of Swingendians in the town
        swing_mean (int): mean of Gaussian distribution to draw from to determine 
            how Swingendian will vote.
        swing_sd (int): sd of Gaussian distribution to draw from to determine 
            how Swingendian will vote.

    Returns:
        (tuple) (swing_big, swing_small) where:
        - swing_big (int): number of Swingendians that vote for Bigendians
        - swing_small (int): number of Swingendians that vote for Smallendians
    """
    swing_big = 0
    swing_small = 0
    
    # For each swingendian choose random from gaussian dist
    for _ in range(num_swingendians):
        side = random.gauss(swing_mean, swing_sd)

        # Add based on side for totals
        if side < 0:
            swing_small += 1
        else:
            swing_big += 1
    return (swing_big, swing_small)


### DO NOT REMOVE THIS LINE ###    
helper.allocate_swingendians_for_town = allocate_swingendians_for_town
###############################

def state_summary(g):
    """
    Summarizes information about the Lilliput graph.
    
    Parameters:
      g (Graph): the graph representation of Lilliput
    
    Returns:
        tuple of six statistics, where the statistics are in order:
            - n_towns (int): number of towns in the Lilliput graph
            - total_voters (int): total number of voters in Lilliput graph
            - big_pct (float): percent of voters who are Bigendian. Value should
                be in the range [0, 100].
            - avg_neighbors (float): the average number of neighbors per town
            - most_big (tuple): the town with the highest margin of Bigendian 
                voters. Represented by (town_name, margin),w here margin = 
                Bigendian voters - Smallendian voters.
            - most_small (tuple): the town with the highest margin of Smallendian
                voters. Represented by (town_name, margin), where margin =
                Bigendian voters - Smallendian voters.
    """
    nodes = g.get_all_nodes()
    n_towns = len(nodes)
    total_voters = 0
    total_big = 0
    total_neighbors = 0
    most_big = None
    most_small = None
    best_big_margin = float('-inf')
    best_small_margin = float('inf')

    # For each town in list of towns
    for town in nodes:

        # Get B and S and calculate margin. Also add to neighbor total
        big, small = town.get_voters_by_party()
        total_voters += big + small
        total_big += big
        total_neighbors += len(g.get_neighbors(town))
        margin = big - small

        # Running track of biggest and smallest margins
        if margin > best_big_margin:
            best_big_margin = margin
            most_big = (town.get_name(), margin)
        if margin < best_small_margin:
            best_small_margin = margin
            most_small = (town.get_name(), margin)

    big_pct = 100.0 * total_big / total_voters
    avg_neighbors = total_neighbors / n_towns

    return (n_towns, total_voters, big_pct, avg_neighbors, most_big, most_small)


def neighbors_of(g, town):
    """
    Finds all the towns that neighbor a particular town. Returns a sorted list 
    of Town objects.
    
    Parameters:
        g (Graph): the graph representation of Lilliput
        town (Town): Town object contained within g
        
    Returns
        list: collection of neighboring Town objects that are sorted by the numeric
            value of the integer town name (e.g. town '2' comes before town '10').
    """
    # Gets all neighbors of town and returns their sorted list
    return sorted(g.get_neighbors(town))


################################################################################
# Part II — ENFORCE DISTRICT CONSTRAINTS with GRAPH ALGORITHMS
################################################################################

def find_shortest_path(g, source, target):
    """
    Finds the shortest path between a `source` town and a `target` town.
    
    Inputs:
        g (Graph): the graph representation of Lilliput
        source (Town): starting Town object
        target (Town): target Town object

    Returns:
        list: all Town objects between source to target (including both source
            and target) if a shortest path exists.
            If no path exists, returns None.
    """
    # Base case if source is already the target
    if source == target:
        return [source]

    # Store parent pointers for path reconstruction
    visited = {source}
    parent = {source: None}
    queue = deque([source])

    # BFS implementation popping from front
    while queue:
        current = queue.popleft()
        for neighbor in g.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current

                # Check if we hit the target and reconstuct path
                if neighbor == target:
                    path = []
                    node = target
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(neighbor)

    return None


def is_compact_connected_subgraph(g, node_set, compactness):
    """
    Determines if a set of towns can qualify to be a legal district based off 
    the connectivity and compactness constraints.
    
    Parameters:
        g (Graph): the graph representation of Towns
        node_set (iterable) : a collection of Town objects to be tested for legal-districthood.
        compactness (int): maximum allowed shortest path length within a district. 
            Path length is defined as the number of towns.
                   
    Returns:
        bool:
            True if all towns within this 'node_set' are connected AND if
                for every pair of towns (A, B) within 'node_set', the shortest path
                between them has length <= compactness.
            False otherwise.
    """
    node_set = set(node_set)

    start = next(iter(node_set))
    visited = {start}
    queue = deque([start])

    # BFS on edges where endpoints in node_set
    while queue:
        current = queue.popleft()

        # Iterate through neighbors of current town
        for nbr in g.get_neighbors(current):
            if nbr in node_set and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    # Ensure set is connected
    if visited != node_set:
        return False

    # Check compactness: shortest path in full graph for every pair
    node_list = list(node_set)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            path = find_shortest_path(g, node_list[i], node_list[j])
            if path is None or len(path) > compactness:
                return False

    return True


################################################################################
# Part III — CONSTRUCT LEGAL DISTRICTS & EVALUATE MAPS
################################################################################

class District(object):
    """
    Represents a set of towns that have already met the legality constraints and 
    requirements. The class keeps track of useful properties.
    
    Define the __init__ of this class with the following attributes:
        self.towns (set): set of Town objects in this district
        self.first_town (Town): The town with the lowest value name (e.g. town 
            '2' has a lower value name than town '10', etc)
        self.big (int): total Bigendian votes across all towns in this district
        self.small (int): total Smallendian votes across all towns in this district
        self.diff (int): self.big - self.small
    """
    def __init__(self, town_list):
        """
        Initialize a District Object
        town_list (list): list of Town objects
        """
        self.towns = set(town_list)
        self.first_town = min(town_list)
        self.big = sum(t.get_voters_by_party()[0] for t in town_list)
        self.small = sum(t.get_voters_by_party()[1] for t in town_list)
        self.diff = self.big - self.small

    def get_towns(self):
        return self.towns

    def get_voters_by_party(self):
        return (self.big, self.small)

    def get_diff(self):
        return self.diff

    def __lt__(self, other):
        return self.first_town < other.first_town

    def __str__(self):
        towns = ''
        for town in sorted(self.towns):
            towns += town.get_name() + ','
        return (f'{towns[:-1]}. ' +
                f'Votes for Big - votes for Small = {self.diff:,}')


def find_all_possible_districts(g, k, compactness):
    """
    Enumerates every possible size-k group of towns and keeps 
    only those that would constitute a legal district. In this pset, we specify that k = 4.
    
    Parameters:
        g (Graph): graph representation of Towns
        k (int): the required number of towns per district 
        compactness (int): The maximum allowed path length between any
            two towns inside a district.     

    Returns:
        (list): collection of legal District objects
    """
    nodes = g.get_all_nodes()
    districts = []

    # Generate all combinations of size k and keep legal ones
    def combine(start, current):
        if len(current) == k:
            if is_compact_connected_subgraph(g, current, compactness):
                districts.append(District(list(current)))
            return
        for i in range(start, len(nodes)):
            current.append(nodes[i])
            combine(i + 1, current)
            current.pop()

    combine(0, [])
    return districts


def find_disjoint_districts(district_list, num_districts):
    """
    Enumerates all full maps (lists of Districts) such that:
        - each map has exactly `num_districts` districts
        - districts in a map are disjoint (no shared Towns)
        - every town appears in a District exactly once
    
    Parameters:
        district_list (list): list of District objects
        num_districts (int): total number of districts required for a valid map

    Returns: 
        (list): A collection of full maps such that each element (map) is:
            - A list of Districts of length num_districts
            - Disjoint (districts have no overlap of towns)
            - All towns are taken into account
    """
    district_list = sorted(district_list)

    all_maps = []

    # Same backtrack pattern as before but now with districs
    def backtrack(start_idx, current_map, used_towns):
        if len(current_map) == num_districts:
            all_maps.append(list(current_map))
            return
        for i in range(start_idx, len(district_list)):
            d = district_list[i]

            # Checking for zero overlap to make disjoint
            if d.get_towns().isdisjoint(used_towns):
                current_map.append(d)
                backtrack(i + 1, current_map, used_towns | d.get_towns())
                current_map.pop()

    backtrack(0, [], set())
    return all_maps


def eval_choices(g, possible_choices):
    """
    Examines a list of valid maps (list of District objects) and evaluates them regarding which party
    benefits from which map
    
    Parameters:
        g (Graph): graph representation of Lilliput
        possible_choices (list): a list of valid maps. Since each map is a list of District objects,
            this is a list of a list of Districts where each inner list is one possible map.


    Returns:
        tuple: (best_big_map, best_small_map, maybe_tie, best_big_num, best_small_num)
        - best_big_map (list): map that maximizes Bigendians's district wins
        - best_small_map (list): map that maximizes Smallendians's district wins
        - maybe_tie (list): any map where there is a tie if one occurs; otherwise None.
        - best_big_num (int): number of districts bigendians would win in best_big_map
        - best_small_num (int): number of districts smallendians would win in best_small_map
    """
    best_big_map = None
    best_small_map = None
    maybe_tie = None
    best_big_num = -1
    best_small_num = -1

    # Iterate through possibilities and count wins
    for m in possible_choices:
        big_wins = 0
        small_wins = 0
        for d in m:
            diff = d.get_diff()
            if diff > 0:
                big_wins += 1
            elif diff < 0:
                small_wins += 1

        # Keep track of best choices by number of wins
        if big_wins > best_big_num:
            best_big_num = big_wins
            best_big_map = m
        if small_wins > best_small_num:
            best_small_num = small_wins
            best_small_map = m
        if big_wins == small_wins:
            maybe_tie = m

    return (best_big_map, best_small_map, maybe_tie, best_big_num, best_small_num)

################################################################################
# Uncomment the code below to run your functions locally
################################################################################

if __name__ == "__main__":
    pass
    ## 1) Build the Lilliput graph
    # g, points, vor = make_lilliput()

    ## 2) Explore graph
    ## Example: get a Town by name string, then list its neighbors (as Towns)
    # t12 = g.get_node("12")
    # nbrs = neighbors_of(g, t12)
    # print("Neighbors of '12':", [t.get_name() for t in nbrs])

    ## 3) Concise printout of state_summary()
    # print(f"Number of towns: {n_towns}")
    # print(f"Total voters statewide: {total_voters:,}")
    # print(f"For entire state, Bigendian receives {big_pct:.2f}% of votes")
    # print(f"Average bordering towns per town: {avg_neighbors:.2f}")
    # if most_big is not None:
    #     print(f"Most Big-leaning town: {most_big[0]} (margin +{most_big[1]:,})")
    # if most_small is not None:
    #     print(f"Most Small-leaning town: {most_small[0]} (margin {most_small[1]:,})")

    ## 4) Run full pipeline (example values)
    # k = 4
    # compactness = 3
    # districts = find_all_possible_districts(g, k, compactness)
    # print("Number of possible districts =", len(districts))
    # maps = find_disjoint_districts(districts, len(g.get_all_nodes()) // k)
    # print("Number of possible combinations of districts =", f"{len(maps):,}")
    # (best_big_map, best_small_map, maybe_tie, best_big_num, best_small_num) = eval_choices(g, maps)

    ## Example plots:
    # plot_graph(g, None, "Connectivity of Towns", None)
