"""Deterministic spatial grid for broad-phase entity queries."""

from __future__ import annotations

from collections import defaultdict

from .arena import MT


class SpatialGrid:
    def __init__(self, cell_size: int = 4 * MT) -> None:
        self.cell_size = cell_size
        self.buckets: dict[tuple[int, int], list] = defaultdict(list)
        self.ordered: list = []
        self.max_collision_radius = 0

    def rebuild(self, entities) -> None:
        self.buckets.clear()
        self.ordered = [entity for entity in entities if entity.alive]
        self.max_collision_radius = max(
            (entity.collision_radius_mt for entity in self.ordered),
            default=0,
        )
        for entity in self.ordered:
            self.buckets[self._cell(entity.pos.x, entity.pos.y)].append(entity)

    def query(self, point, radius: int) -> list:
        if radius < 0:
            return []
        cell = self.cell_size
        min_x = (point.x - radius) // cell
        max_x = (point.x + radius) // cell
        min_y = (point.y - radius) // cell
        max_y = (point.y + radius) // cell
        found = set()
        for grid_x in range(min_x, max_x + 1):
            for grid_y in range(min_y, max_y + 1):
                found.update(id(entity) for entity in
                             self.buckets.get((grid_x, grid_y), ()))
        # Preserve the entity-table order used by the legacy loop. This keeps
        # equal-distance tie-breaking identical.
        return [entity for entity in self.ordered if id(entity) in found]

    def _cell(self, x: int, y: int) -> tuple[int, int]:
        return x // self.cell_size, y // self.cell_size
