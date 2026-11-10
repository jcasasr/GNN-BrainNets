import networkx as nx
import numpy as np
import sys
import logging


# Print some basics properties of G
def report_graph_basics(G):
    A = nx.to_numpy_array(G)
    v_min = np.min(A)
    v_max = np.max(A)
    n_zero = np.count_nonzero(A==0)
    n_non_zero = np.count_nonzero(A)

    logging.debug("   Min={:.2f}, Max={:.2f}, # Zero={}, # Non-Zero={}".format(v_min, v_max, n_zero, n_non_zero))


def create_graph_from_AM(A):
    """
    Create graph G from adjacency matrix A
    :param A: Adjacency matrix
    :return: Graph G
    """
    # create undirected and weighted graph from adjacency matrix
    G = nx.from_numpy_array(A=A, parallel_edges=False)

    return G


def create_distance_A_from_A(A):
    """
    Compute the 1/w to obtain the "distance" between nodes 
    higher values mean shorter distance
    :param A: ndarray Adjacency matrix (152x152)
    :return: ndarray Inverse adjacency matrix (152x152)
    """
    A_inv = np.zeros(A.shape, dtype=float)

    # Compte A=1/w 
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not A[i,j] == 0:
                A_inv[i,j] = 1 / A[i,j]

    return A_inv


def create_distance_graph_from_AM(A):
    """
    some measures, like betweenness and closeness centrality, are based on distance, 
    so we need to compute 1/weights in order to get the shortest paths
    """
    # Compte A=1/w 
    A_inv = create_distance_A_from_A(A)

    # create undirected and weighted graph from adjacency matrix
    G = create_graph_from_AM(A_inv)

    return G


def compute_A_min(A):
    """
    Compute the matrix of minimum distances between nodes considering the multilayer network

    Parameters
    ----------
    A : ndarray
        Super-Adjacency matrix (152x152)
    
    Returns
    -------
    ndarray
        Adjacency matrix (76x76)
    """
    # number of nodes in supergraph
    num_nodes_SG = A.shape[0]

    # number of nodes in single graph
    if (num_nodes_SG % 2) == 0:
        num_nodes_G = int(num_nodes_SG / 2)
    else:
        logging.error("The number of nodes in super graph is not EVEN! ({})".format(num_nodes_SG))
        sys.exit(0)

    ###
    # Step 1: create A_min
    # A_min[i,j]: minimum path from i to j through any layer
    A_min = np.zeros((num_nodes_G, num_nodes_G), dtype=float)

    ### DEBUG
    count_min = np.zeros(4, dtype=int)

    # for each node in G (i.e. [1..76])
    for source in range(num_nodes_G):
        for target in range(num_nodes_G):
            # distance value through different layers
            tmp = np.asarray((A[source, target] # source RS --> target RS
                , A[source, (target+num_nodes_G)] # source RS --> target GM (through FA)
                , A[source+num_nodes_G, target] # source GM --> target RS (through FA)
                , A[(source+num_nodes_G), (target+num_nodes_G)])) # source GM --> target GM

            if np.count_nonzero(tmp):
                min_value = np.min(tmp[np.nonzero(tmp)])
                A_min[source, target] = min_value

                ### DEBUG 
                ind = np.where(tmp==min_value)[0][0]
                count_min[ind] = count_min[ind] + 1

            else:
                A_min[source, target] = 0

    ### DEBUG
    count_min_total = np.sum(count_min)
    count_min_rs = count_min[0]
    count_min_fa = count_min[1] + count_min[2] # FA includes: RS -> GM and GM -> RS
    count_min_gm = count_min[3]
    logging.info("RS={} ({:.2f} %), FA={} ({:.2f} %), GM={} ({:.2f} %), Total={}".format(
        count_min_rs, (count_min_rs / count_min_total) * 100
        , count_min_fa, (count_min_fa / count_min_total) * 100
        , count_min_gm, (count_min_gm / count_min_total) * 100
        , count_min_total))

    return A_min


def compute_CC_ML(G):
    """
    Compute the closeness centrality (CC) in multilayer networks

    PARAMETERS
    ----------

    G: Networkx graph
        Adjancency matrix of minum distances between nodes (76x76)
    """
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())

    # array to store the scores
    values = np.zeros(num_nodes, dtype=float)

    # compute SP for all nodes
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    
    # for each node, compute the CC
    # CC(x) = (N-1) / (Sum_y d(x,y))
    for node in nodes:
        values[node] = (num_nodes-1) / sum(sp[node].values())

    return values


def compute_LE_SL(G):
    """
    Local Efficiency (LE) for single layer networks   

    PARAMETERS
    ----------

    G: Networkx graph
        Adjancency matrix of minum distances between nodes (76x76) 
    """
    num_nodes = G.number_of_nodes()

    # array to store the scores
    values = np.zeros(num_nodes, dtype=float)

    # for each node in G
    for node in list(G.nodes):
        # get the ego network
        subG = nx.ego_graph(G, n=node, radius=1, center=False, undirected=True)
        logging.debug("Creating subgraph of 1-neighbourhood of node {}... [n={} and e={}]".format(node, subG.number_of_nodes(), subG.number_of_edges()))

        # node efficiency
        eff_node = 0

        # check if subgraph is not null
        if subG.number_of_nodes() > 1:
            # check if it is connected
            if not nx.is_connected(subG):
                logging.debug("   Subgraph is not connected: extracting the largest component...")
                subG = subG.subgraph(max(nx.connected_components(subG), key=len))
                logging.debug("   New subgraph of node {} has n={} and e={}".format(node, subG.number_of_nodes(), subG.number_of_edges()))

            # compute SP for all nodes in ego network
            sp = dict(nx.all_pairs_dijkstra_path_length(subG, weight='weight'))
            subnodes = list(subG.nodes())
            num_subnodes = subG.number_of_nodes()

            for node_source in subnodes:
                for node_target in subnodes:
                    if(node_source != node_target):
                        eff_node = eff_node + (1 / sp[node_source][node_target])
            
            if num_subnodes > 1:
                # normalize by number of nodes in subgraph
                eff_node = eff_node / (num_subnodes * (num_subnodes - 1))
            else:
                eff_node = 0
        
        # store node efficiency
        values[node] = eff_node

    return values