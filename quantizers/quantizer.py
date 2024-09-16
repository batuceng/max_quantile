import numpy as np
import torch

class GridQuantizer(nn.Module):
    def __init__(self, y_vals, proto_count_per_dim):
        super(GridQuantizer, self).__init__()
        self.mins = np.min(y_vals, axis=0)  # (d,)
        self.maxs = np.max(y_vals, axis=0)  # (d,)
        d = y_vals.shape[1]  # Number of dimensions

        if isinstance(proto_count_per_dim, int):
            proto_count_per_dim = [proto_count_per_dim] * d  # Make it a list of length d
        assert len(proto_count_per_dim) == d, "proto_count_per_dim must be of length d or a single integer"

        boundaries = [np.linspace(self.mins[i], self.maxs[i], proto_count_per_dim[i]+1) for i in range(d)]  # List of (proto_count_per_dim[i],) for each dimension
        grids = [[np.mean([up,down]) for up,down in zip(bounds[1:],bounds[:-1])] for bounds in boundaries] 
        
        grid = np.meshgrid(*grids)  # This creates a grid for each dimension
        protos_numpy = np.vstack([g.ravel() for g in grid]).T  # (k^d, d), where k varies per dimension based on proto_count_per_dim
        
        self.protos = torch.tensor(protos_numpy, dtype=torch.float32, requires_grad=False)
        
    def quantize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        distances = torch.norm(self.protos - x, dim=1)  # Compute distances in PyTorch
        nearest_idx = torch.argmin(distances)  # Get the index of the nearest prototype
        return self.protos[nearest_idx]  # Return the nearest prototype


# Placeholder class for Voronoi Quantizer
class VoronoiQuantizer(GridQuantizer):
    def __init__(self, y_vals, proto_count_per_dim):
        super(VoronoiQuantizer, self).__init__(y_vals, proto_count_per_dim)
        
    def quantize(self, x):
        return super(VoronoiQuantizer, self).quantize(x)

    # Delete the prototypes in the given indices
    def remove_proto(self, indices):
        mask = np.full(len(self.protos),True,dtype=bool)
        mask[indices] = False
        self.protos = self.protos[mask]

    # Repeat the prototypes in the given indices and concat to the end
    def add_proto(self, indices):
        mask = np.full(len(self.protos),False,dtype=bool)
        mask[indices] = True
        self.protos = torch.vstack(self.protos, self.protos[mask])

    def get_protos(self):
        return self.protos

    def set_protos(self, protos):
        self.protos = protos
    
    def get_voronoi_areas(self):
        pass
    
 
def QuadTree():
    def __init__():
        pass