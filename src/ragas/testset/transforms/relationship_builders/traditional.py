import itertools
import typing as t
from collections import Counter
from dataclasses import dataclass

from ragas.metrics.collections._string import DistanceMeasure
from ragas.testset.graph import KnowledgeGraph, Node, Relationship
from ragas.testset.transforms.base import RelationshipBuilder
from ragas.utils import rapidfuzz_distances


@dataclass
class JaccardSimilarityBuilder(RelationshipBuilder):
    property_name: str = "entities"
    key_name: t.Optional[str] = None
    new_property_name: str = "jaccard_similarity"
    threshold: float = 0.5

    def _jaccard_similarity(self, set1: t.Set[str], set2: t.Set[str]) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _find_similar_embedding_pairs(
        self, kg: KnowledgeGraph
    ) -> t.List[t.Tuple[int, int, float]]:
        """
        Finds all node index pairs with Jaccard similarity above the threshold.
        Returns a set of (i, j, similarity) tuples.
        """

        similar_pairs = set()
        for (i, node1), (j, node2) in itertools.combinations(enumerate(kg.nodes), 2):
            items1 = node1.get_property(self.property_name)
            items2 = node2.get_property(self.property_name)
            if items1 is None or items2 is None:
                raise ValueError(
                    f"Node {node1.id} or {node2.id} has no {self.property_name}"
                )
            if self.key_name is not None:
                items1 = items1.get(self.key_name, [])
                items2 = items2.get(self.key_name, [])
            similarity = self._jaccard_similarity(set(items1), set(items2))
            if similarity >= self.threshold:
                similar_pairs.add((i, j, similarity))
        return list(similar_pairs)

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        similar_pairs = self._find_similar_embedding_pairs(kg)
        return [
            Relationship(
                source=kg.nodes[i],
                target=kg.nodes[j],
                type=self.new_property_name,
                properties={self.new_property_name: similarity_float},
                bidirectional=True,
            )
            for i, j, similarity_float in similar_pairs
        ]


@dataclass
class OverlapScoreBuilder(RelationshipBuilder):
    property_name: str = "entities"
    key_name: t.Optional[str] = None
    new_property_name: str = "overlap_score"
    distance_measure: DistanceMeasure = DistanceMeasure.JARO_WINKLER
    distance_threshold: float = 0.9
    threshold: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        rapidfuzz_distances(DistanceMeasure)  # rapidfuzz is installed

    @property
    def distance_measure_map(self) -> t.Dict[DistanceMeasure, t.Any]:
        return rapidfuzz_distances(DistanceMeasure)

    def _overlap_score(self, overlaps: t.List[bool]) -> float:
        return sum(overlaps) / len(overlaps) if len(overlaps) > 0 else 0.0

    def _get_noisy_items(
        self, nodes: t.List[Node], property_name: str, percent_cut_off: float = 0.05
    ) -> t.List[str]:
        all_items = []
        for node in nodes:
            items = node.get_property(property_name)
            if items is not None:
                if isinstance(items, str):
                    all_items.append(items)
                elif isinstance(items, list):
                    all_items.extend(items)
                else:
                    pass

        num_unique_items = len(set(all_items))
        num_noisy_items = max(1, int(num_unique_items * percent_cut_off))
        noisy_list = list(dict(Counter(all_items).most_common()).keys())[
            :num_noisy_items
        ]
        return noisy_list

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        distance_measure = self.distance_measure_map[self.distance_measure]
        noisy_items = self._get_noisy_items(kg.nodes, self.property_name)
        relationships = []
        for i, node_x in enumerate(kg.nodes):
            for j, node_y in enumerate(kg.nodes):
                if i >= j:
                    continue
                node_x_items = node_x.get_property(self.property_name)
                node_y_items = node_y.get_property(self.property_name)
                if node_x_items is None or node_y_items is None:
                    raise ValueError(
                        f"Node {node_x.id} or {node_y.id} has no {self.property_name}"
                    )
                if self.key_name is not None:
                    node_x_items = node_x_items.get(self.key_name, [])
                    node_y_items = node_y_items.get(self.key_name, [])

                overlaps = []
                overlapped_items = []
                for x in node_x_items:
                    if x not in noisy_items:
                        for y in node_y_items:
                            if y not in noisy_items:
                                similarity = 1 - distance_measure.distance(
                                    x.lower(), y.lower()
                                )
                                verdict = similarity >= self.distance_threshold
                                overlaps.append(verdict)
                                if verdict:
                                    overlapped_items.append((x, y))

                similarity = self._overlap_score(overlaps)
                if similarity >= self.threshold:
                    relationships.append(
                        Relationship(
                            source=node_x,
                            target=node_y,
                            type=f"{self.property_name}_overlap",
                            properties={
                                f"{self.property_name}_{self.new_property_name}": similarity,
                                "overlapped_items": overlapped_items,
                            },
                            bidirectional=True,
                        )
                    )

        return relationships
