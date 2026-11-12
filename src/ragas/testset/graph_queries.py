import hashlib
import random
import typing as t

from ragas.testset.graph import KnowledgeGraph, Node, Relationship


def get_child_nodes(node: Node, graph: KnowledgeGraph, level: int = 1) -> t.List[Node]:
    """
    Get the child nodes of a given node up to a specified level.

    Parameters
    ----------
    node : Node
        The node to get the children of.
    graph : KnowledgeGraph
        The knowledge graph containing the node.
    level : int
        The maximum level to which child nodes are searched.

    Returns
    -------
    List[Node]
        The list of child nodes up to the specified level.
    """
    children = []

    # Helper function to perform depth-limited search for child nodes
    def dfs(current_node: Node, current_level: int):
        if current_level > level:
            return
        for rel in graph.relationships:
            if rel.source == current_node and rel.type == "child":
                children.append(rel.target)
                dfs(rel.target, current_level + 1)

    # Start DFS from the initial node at level 0
    dfs(node, 1)

    return children


def get_parent_nodes(node: Node, graph: KnowledgeGraph, level: int = 1) -> t.List[Node]:
    """
    Get the parent nodes of a given node up to a specified level.

    Parameters
    ----------
    node : Node
        The node to get the parents of.
    graph : KnowledgeGraph
        The knowledge graph containing the node.
    level : int
        The maximum level to which parent nodes are searched.

    Returns
    -------
    List[Node]
        The list of parent nodes up to the specified level.
    """
    parents = []

    # Helper function to perform depth-limited search for parent nodes
    def dfs(current_node: Node, current_level: int):
        if current_level > level:
            return
        for rel in graph.relationships:
            if rel.target == current_node and rel.type == "child":
                parents.append(rel.source)
                dfs(rel.source, current_level + 1)

    # Start DFS from the initial node at level 0
    dfs(node, 1)

    return parents


def find_n_indirect_clusters(
    graph: KnowledgeGraph,
    n: int,
    relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
    depth_limit: int = 3,
) -> t.List[t.Set[Node]]:
    """
    Return n indirect clusters of nodes in the knowledge graph based on a relationship condition.
    Optimized for large datasets by using an adjacency index for lookups and limiting path exploration
    relative to n.

    A cluster represents a path through the graph. For example, if A -> B -> C -> D exists in the graph,
    then {A, B, C, D} forms a cluster. If there's also a path A -> B -> C -> E, it forms a separate cluster.

    The method returns a list of up to n sets, where each set contains nodes forming a complete path
    from a starting node to a leaf node or a path segment up to depth_limit nodes long. The result may contain
    fewer than n clusters if the graph is very sparse or if there aren't enough nodes to form n distinct clusters.

    To maximize diversity in the results:
    1. Random starting nodes are selected
    2. Paths from each starting node are grouped
    3. Clusters are selected in round-robin fashion from each group until n unique clusters are found
    4. Duplicate clusters are eliminated
    5. When a superset cluster is found (e.g., {A,B,C,D}), any existing subset clusters (e.g., {A,B,C})
       are removed to avoid redundancy

    Parameters
    ----------
    graph : KnowledgeGraph
        The knowledge graph to search.
    n : int
        Target number of clusters to return. Must be at least 1. Should return n clusters unless the graph is
        extremely sparse.
    relationship_condition : Callable[[Relationship], bool], optional
        A function that takes a Relationship and returns a boolean, by default lambda _: True
    depth_limit : int, optional
        Maximum depth for path exploration, by default 3. Must be at least 2 to form clusters by definition.

    Returns
    -------
    List[Set[Node]]
        A list of sets, where each set contains nodes that form a cluster.

    Raises
    ------
    ValueError
        If depth_limit < 2, n < 1, or no relationships match the provided condition.
    """
    if depth_limit < 2:
        raise ValueError("depth_limit must be at least 2 to form valid clusters")

    if n < 1:
        raise ValueError("n must be at least 1")

    # Filter relationships once upfront
    filtered_relationships: list[Relationship] = [
        rel for rel in graph.relationships if relationship_condition(rel)
    ]

    if not filtered_relationships:
        raise ValueError(
            "No relationships match the provided condition. Cannot form clusters."
        )

    # Build adjacency list for faster neighbor lookup - optimized for large datasets
    adjacency_list: dict[Node, set[Node]] = {}
    unique_edges: set[frozenset[Node]] = set()
    for rel in filtered_relationships:
        # Lazy initialization since we only care about nodes with relationships
        if rel.source not in adjacency_list:
            adjacency_list[rel.source] = set()
        adjacency_list[rel.source].add(rel.target)
        unique_edges.add(frozenset({rel.source, rel.target}))
        if rel.bidirectional:
            if rel.target not in adjacency_list:
                adjacency_list[rel.target] = set()
            adjacency_list[rel.target].add(rel.source)

    # Aggregate clusters for each start node
    start_node_clusters: dict[Node, set[frozenset[Node]]] = {}
    # sample enough starting nodes to handle worst case grouping scenario where nodes are grouped
    # in independent clusters of size equal to depth_limit. This only surfaces when there are less
    # unique edges than nodes.
    connected_nodes: set[Node] = set().union(*unique_edges)
    sample_size: int = (
        (n - 1) * depth_limit + 1
        if len(unique_edges) < len(connected_nodes)
        else max(n, depth_limit, 10)
    )

    def dfs(node: Node, start_node: Node, current_path: t.Set[Node]):
        # Terminate exploration when max usable clusters is reached so complexity doesn't spiral
        if len(start_node_clusters.get(start_node, [])) > sample_size:
            return

        current_path.add(node)
        path_length = len(current_path)
        at_max_depth = path_length >= depth_limit
        neighbors = adjacency_list.get(node, None)

        # If this is a leaf node or we've reached depth limit
        # and we have a valid path of at least 2 nodes, add it as a cluster
        if path_length > 1 and (
            at_max_depth or not neighbors or all(n in current_path for n in neighbors)
        ):
            # Lazy initialization of the set for this start node
            if start_node not in start_node_clusters:
                start_node_clusters[start_node] = set()
            start_node_clusters[start_node].add(frozenset(current_path))
        elif neighbors:
            for neighbor in neighbors:
                # Block cycles
                if neighbor not in current_path:
                    dfs(neighbor, start_node, current_path)

        # Backtrack by removing the current node from path
        current_path.remove(node)

    # Shuffle nodes for random starting points
    # Use adjacency list since that has filtered out isolated nodes
    # Sort by node ID for consistent ordering while maintaining algorithm effectiveness
    start_nodes = sorted(adjacency_list.keys(), key=lambda n: n.id.hex)
    # Use a hash-based seed for reproducible but varied shuffling based on the nodes themselves
    node_ids_str = "".join(n.id.hex for n in start_nodes)
    node_hash = hashlib.sha256(node_ids_str.encode("utf-8")).hexdigest()
    rng = random.Random(int(node_hash[:8], 16))  # Use first 8 hex chars as seed
    rng.shuffle(start_nodes)
    samples: list[Node] = start_nodes[:sample_size]
    for start_node in samples:
        dfs(start_node, start_node, set())

    start_node_clusters_list: list[set[frozenset[Node]]] = list(
        start_node_clusters.values()
    )

    # Iteratively pop from each start_node_clusters until we have n unique clusters
    # Avoid adding duplicates and subset/superset pairs so we have diversity. We
    # favor supersets over subsets if we are given a choice.
    unique_clusters: set[frozenset[Node]] = set()
    i = 0
    while len(unique_clusters) < n and start_node_clusters_list:
        # Cycle through the start node clusters
        current_index = i % len(start_node_clusters_list)

        current_start_node_clusters: set[frozenset[Node]] = start_node_clusters_list[
            current_index
        ]
        cluster: frozenset[Node] = current_start_node_clusters.pop()

        # Check if the new cluster is a subset of any existing cluster
        # and collect any existing clusters that are subsets of this cluster
        is_subset = False
        subsets_to_remove: set[frozenset[Node]] = set()

        for existing in unique_clusters:
            if cluster.issubset(existing):
                is_subset = True
                break
            elif cluster.issuperset(existing):
                subsets_to_remove.add(existing)

        # Only add the new cluster if it's not a subset of any existing cluster
        if not is_subset:
            # Remove any subsets of the new cluster
            unique_clusters -= subsets_to_remove
            unique_clusters.add(cluster)

        # If this set is now empty, remove it
        if not current_start_node_clusters:
            start_node_clusters_list.pop(current_index)
            # Don't increment i since we removed an element to account for shift
        else:
            i += 1

    return [set(cluster) for cluster in unique_clusters]


def find_two_nodes_single_rel(
    graph: KnowledgeGraph,
    relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
) -> t.List[t.Tuple[Node, Relationship, Node]]:
    """
    Finds nodes in the knowledge graph based on a relationship condition.
    (NodeA, NodeB, Rel) triples are considered as multi-hop nodes.

    Parameters
    ----------
    graph : KnowledgeGraph
        The knowledge graph to search.
    relationship_condition : Callable[[Relationship], bool], optional
        A function that takes a Relationship and returns a boolean, by default lambda _: True

    Returns
    -------
    List[Set[Node, Relationship, Node]]
        A list of sets, where each set contains two nodes and a relationship forming a multi-hop node.
    """

    relationships = [
        relationship
        for relationship in graph.relationships
        if relationship_condition(relationship)
    ]

    triplets = set()

    for relationship in relationships:
        if relationship.source != relationship.target:
            node_a = relationship.source
            node_b = relationship.target
            # Ensure the smaller ID node is always first
            if node_a.id < node_b.id:
                normalized_tuple = (node_a, relationship, node_b)
            else:
                normalized_relationship = Relationship(
                    source=node_b,
                    target=node_a,
                    type=relationship.type,
                    properties=relationship.properties,
                )
                normalized_tuple = (node_b, normalized_relationship, node_a)

            triplets.add(normalized_tuple)

    return list(triplets)
