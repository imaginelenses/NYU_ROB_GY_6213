import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class EDT:
    def __init__(self, wall_corner_list, cell_size=0.1, alpha=5.0):
        self.wall_corner_list = wall_corner_list
        self.cell_size = cell_size
        self.alpha = alpha

        self.x_min = min(min(w[0], w[2]) for w in wall_corner_list)
        self.x_max = max(max(w[0], w[2]) for w in wall_corner_list)
        self.y_min = min(min(w[1], w[3]) for w in wall_corner_list)
        self.y_max = max(max(w[1], w[3]) for w in wall_corner_list)

        self.x_cells = int(np.ceil((self.x_max - self.x_min) / cell_size))
        self.y_cells = int(np.ceil((self.y_max - self.y_min) / cell_size))

        self.edt_grid = self._build_edt()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_edt(self):
        occupancy = np.ones((self.y_cells, self.x_cells), dtype=np.float32)

        for w in self.wall_corner_list:
            wx0, wy0, wx1, wy1 = w
            length = np.hypot(wx1 - wx0, wy1 - wy0)
            n = max(int(length / (self.cell_size * 0.5)), 2)
            xs = np.linspace(wx0, wx1, n)
            ys = np.linspace(wy0, wy1, n)
            ci, cj = self._to_grid_arrays(xs, ys)
            occupancy[cj, ci] = 0.0

        # Mask exterior: all cells outside the room polygon are treated as walls
        polygon_pts = [(w[0], w[1]) for w in self.wall_corner_list]
        room_path = MplPath(polygon_pts + [polygon_pts[0]])
        xs_grid = self.x_min + (np.arange(self.x_cells) + 0.5) * self.cell_size
        ys_grid = self.y_min + (np.arange(self.y_cells) + 0.5) * self.cell_size
        xx, yy = np.meshgrid(xs_grid, ys_grid)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = room_path.contains_points(pts).reshape(self.y_cells, self.x_cells)
        occupancy[~inside] = 0.0

        return distance_transform_edt(occupancy) * self.cell_size
    
    def _build_graph(self):
        N = self.y_cells * self.x_cells
        flat_edt = self.edt_grid.ravel()

        # All cell indices as 2-D coordinate arrays
        j_all, i_all = np.mgrid[0:self.y_cells, 0:self.x_cells]
        i_flat = i_all.ravel()
        j_flat = j_all.ravel()
        src_valid = flat_edt > 0.0

        moves = [
            ( 1,  0, self.cell_size),
            (-1,  0, self.cell_size),
            ( 0,  1, self.cell_size),
            ( 0, -1, self.cell_size),
            ( 1,  1, self.cell_size * np.sqrt(2)),
            ( 1, -1, self.cell_size * np.sqrt(2)),
            (-1,  1, self.cell_size * np.sqrt(2)),
            (-1, -1, self.cell_size * np.sqrt(2)),
        ]

        rows_list, cols_list, data_list = [], [], []
        for di, dj, dist in moves:
            ni = i_flat + di
            nj = j_flat + dj
            in_bounds = (ni >= 0) & (ni < self.x_cells) & (nj >= 0) & (nj < self.y_cells)
            ni_c = np.clip(ni, 0, self.x_cells - 1)
            nj_c = np.clip(nj, 0, self.y_cells - 1)
            mask = src_valid & in_bounds & (self.edt_grid[nj_c, ni_c] > 0.0)

            src_idx = j_flat[mask] * self.x_cells + i_flat[mask]
            dst_idx = nj_c[mask]  * self.x_cells + ni_c[mask]
            src_clearance = flat_edt[src_idx]
            dst_clearance = self.edt_grid[nj_c[mask], ni_c[mask]]
            weights = self._edge_weight(dist, src_clearance, dst_clearance)
            rows_list.append(src_idx)
            cols_list.append(dst_idx)
            data_list.append(weights)

        return csr_matrix(
            (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
            shape=(N, N),
        )

    # ------------------------------------------------------------------
    # Internal vectorised helpers (no clipping guard exposed yet)
    # ------------------------------------------------------------------

    def _to_grid_arrays(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        i = np.clip(
            ((x - self.x_min) / self.cell_size).astype(int), 0, self.x_cells - 1
        )
        j = np.clip(
            ((y - self.y_min) / self.cell_size).astype(int), 0, self.y_cells - 1
        )
        return i, j

    def _distance_to_boundary(self, i, j):
        """Return EDT value (metres) for grid indices i, j.

        Accepts scalars or numpy arrays; uses vectorised fancy indexing.
        """
        i = np.clip(np.asarray(i, dtype=int), 0, self.x_cells - 1)
        j = np.clip(np.asarray(j, dtype=int), 0, self.y_cells - 1)
        return self.edt_grid[j, i]

    def _cost(self, dis):
        """Exponential decay cost from boundary.

        cost = exp(-d / alpha)
        -> 1.0  at the wall  (d = 0)
        -> 0.0  far from walls (d → ∞)
        alpha controls the decay rate:
        small alpha → sharp falloff, only penalizes cells very close to walls
        large alpha → gradual falloff, penalizes cells further from walls too
        Accepts scalars or numpy arrays.
        """
        return np.exp(-np.asarray(dis, dtype=float) / self.alpha)

    def _edge_weight(self, step_distance, src_clearance, dst_clearance):
        """Convert wall proximity into a traversal weight for graph search.

        The shortest-path graph should be driven by the same wall-clearance field
        shown in the heat map. Using the narrowest clearance along an edge makes
        paths back away from corners instead of only optimizing Euclidean length.
        """
        clearance = np.minimum(np.asarray(src_clearance, dtype=float), np.asarray(dst_clearance, dtype=float))
        wall_cost = self._cost(clearance)
        return step_distance / np.clip(1.0 - wall_cost, 1e-6, None)
    
    # ------------------------------------------------------------------
    # Public API — all fully vectorised via numpy
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices.

        Accepts scalars or numpy arrays.
        Returns (i, j) where i = column (x-axis), j = row (y-axis).
        """
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        i, j = self._to_grid_arrays(x, y)
        if scalar:
            return int(i), int(j)
        return i, j

    def get_cost(self, x, y):
        i, j = self.world_to_grid(x, y)
        dis = self._distance_to_boundary(i, j)
        return self._cost(dis)

    def shortest_path(self, start_x, start_y, goal_x, goal_y):
        start_i, start_j = self.world_to_grid(start_x, start_y)
        goal_i, goal_j = self.world_to_grid(goal_x, goal_y)

        start_node = start_j * self.x_cells + start_i
        goal_node  = goal_j  * self.x_cells + goal_i

        _, predecessors = dijkstra(
            self._graph, directed=True, indices=start_node, return_predecessors=True
        )

        if predecessors[goal_node] == -9999:
            return None

        path_nodes = []
        node = goal_node
        while node != start_node:
            path_nodes.append(node)
            node = predecessors[node]
            if node == -9999:
                return None
        path_nodes.append(start_node)
        path_nodes.reverse()

        path_nodes = np.array(path_nodes)
        pi = path_nodes % self.x_cells
        pj = path_nodes // self.x_cells
        wx = self.x_min + (pi + 0.5) * self.cell_size
        wy = self.y_min + (pj + 0.5) * self.cell_size
        return list(zip(wx, wy))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, path=None, figsize=(14, 6)):
        """Plot the EDT and cost field side by side, with an optional path overlay."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # Use the actual grid extent (ceil may add a partial cell beyond x_max/y_max),
        # so that each pixel is exactly cell_size wide/tall and walls align correctly.
        extent = [
            self.x_min, self.x_min + self.x_cells * self.cell_size,
            self.y_min, self.y_min + self.y_cells * self.cell_size,
        ]

        im0 = axes[0].imshow(
            self.edt_grid, origin="lower", extent=extent, cmap="plasma"
        )
        self._draw_walls(axes[0], color="white")
        plt.colorbar(im0, ax=axes[0], label="Distance to boundary (m)")
        axes[0].set_title("Euclidean Distance Transform")
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")

        cost_grid = self._cost(self.edt_grid)
        im1 = axes[1].imshow(
            cost_grid, origin="lower", extent=extent, cmap="hot_r"
        )
        self._draw_walls(axes[1], color="white")
        plt.colorbar(im1, ax=axes[1], label="Cost")
        axes[1].set_title(f"Cost field (alpha={self.alpha})")
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")

        if path is not None:
            px, py = zip(*path)
            for ax in axes:
                ax.plot(px, py, color="cyan", linewidth=1.5, zorder=3)
                ax.plot(px[0],  py[0],  "go", markersize=6, zorder=4)
                ax.plot(px[-1], py[-1], "rs", markersize=6, zorder=4)

        plt.tight_layout()
        plt.show()

    def _draw_walls(self, ax, color="black", lw=1):
        for w in self.wall_corner_list:
            ax.plot([w[0], w[2]], [w[1], w[3]], color=color, linewidth=lw)


# ----------------------------------------------------------------------
# Quick smoke-test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import parameters
    wall_corner_list = parameters.wall_corner_list
    
    edt = EDT(wall_corner_list, cell_size=0.1, alpha=parameters.mppi_alpha)

    # Scalar lookup
    i, j = edt.world_to_grid(0.0, 0.0)
    print(f"world (0,0) -> grid ({i},{j})")
    print(f"distance_to_boundary: {edt._distance_to_boundary(i, j):.3f} m")
    print(f"cost:                 {edt._cost(edt._distance_to_boundary(i, j)):.4f}")

    # Vectorised lookup over a grid of world points
    xs = np.linspace(edt.x_min + 0.5, edt.x_max - 0.5, 5)
    ys = np.linspace(edt.y_min + 0.5, edt.y_max - 0.5, 5)
    gx, gy = np.meshgrid(xs, ys)
    gi, gj = edt.world_to_grid(gx.ravel(), gy.ravel())
    dists = edt._distance_to_boundary(gi, gj)
    costs = edt._cost(dists)
    print("\nVectorised distances (sample):", np.round(dists, 3))
    print("Vectorised costs    (sample):", np.round(costs, 4))

    # Shortest path between two interior points
    path = edt.shortest_path(0.0, 0.0, 1.5, -1.5)
    if path is None:
        print("\nShortest path: None (start or goal on a wall cell)")
    else:
        print(f"\nShortest path: {len(path)} waypoints")
        for wp in path[:5]:
            print(f"  ({wp[0]:.2f}, {wp[1]:.2f})")
        if len(path) > 5:
            print(f"  ... ({len(path) - 5} more)")

    edt.visualize(path=path)