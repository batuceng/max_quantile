import numpy as np
import torch
import torch.nn as nn
from itertools import product

from .voronoi import get_voronoi_areas
from scipy.spatial import ConvexHull

class GridQuantizer(nn.Module):
    def __init__(self, y_vals, proto_count_per_dim):
        super(GridQuantizer, self).__init__()
        self.mins = np.min(y_vals, axis=0)  # (d,)
        self.maxs = np.max(y_vals, axis=0)  # (d,)
        self.outer_hull = self.get_data_boundary_box(self.mins, self.maxs)
        d = y_vals.shape[1]  # Number of dimensions
        self.dims = d

        if isinstance(proto_count_per_dim, int):
            proto_count_per_dim = [proto_count_per_dim] * d  # Make it a list of length d
        assert len(proto_count_per_dim) == d, "proto_count_per_dim must be of length d or a single integer"

        proto_boundaries = [np.linspace(self.mins[i], self.maxs[i], proto_count_per_dim[i]+1) for i in range(d)]  # List of (proto_count_per_dim[i],) for each dimension
        grids = [[np.mean([up,down]) for up,down in zip(bounds[1:],bounds[:-1])] for bounds in proto_boundaries] 
        
        grid = np.meshgrid(*grids)  # This creates a grid for each dimension
        protos_numpy = np.vstack([g.ravel() for g in grid]).T  # (k^d, d), where k varies per dimension based on proto_count_per_dim
        
        self.protos = nn.Parameter(torch.tensor(protos_numpy, dtype=torch.float32), requires_grad=False)
        
    def quantize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        cdist_list = torch.cdist(x, self.protos, p=2)
        mindist, pos = torch.min(cdist_list, dim=1)
        return mindist, pos  # Return the index of nearest prototype

    def get_data_boundary_box(self, mins, maxs):
        # Create a list of tuples with min and max for each dimension
        bounds = [(mins[i], maxs[i]) for i in range(len(mins))]
        
        # Generate all possible combinations of mins and maxs (2^d corner points)
        corner_points = np.array(list(product(*bounds)))
        
        return ConvexHull(corner_points)
class VoronoiQuantizer(GridQuantizer):
    def __init__(self, y_vals, proto_count_per_dim):
        super(VoronoiQuantizer, self).__init__(y_vals, proto_count_per_dim)
        self.protos.requires_grad = True
        
    def quantize(self, x):
        return super(VoronoiQuantizer, self).quantize(x)

    # Delete the prototypes in the given indices
    @torch.no_grad()
    def remove_proto(self, indices):
        mask = np.full(len(self.protos),True,dtype=bool)
        mask[indices] = False
        self.protos = self.protos[mask]

    # Repeat the prototypes in the given indices and concat to the end
    @torch.no_grad()
    def add_proto(self, indices):
        mask = np.full(len(self.protos),False,dtype=bool)
        mask[indices] = True
        self.protos = torch.vstack(self.protos, self.protos[mask])
    # Return area of each proto
    def get_areas(self):
        outer_point_list = self.outer_hull.points[self.outer_hull.vertices]
        return get_voronoi_areas(self.protos.detach().cpu().numpy(), outer_point_list)
    
    
def QuadTree():
    def __init__():
        pass