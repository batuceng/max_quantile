import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import warnings


from .voronoi import get_voronoi_areas, get_voronoi_boundaries
from scipy.spatial import ConvexHull, Voronoi
import pyvoro

class GridQuantizer(nn.Module):
    def __init__(self, y_vals, proto_count_per_dim):
        super(GridQuantizer, self).__init__()
        self.mins = np.min(y_vals , axis=0)  # (d,)
        self.maxs = np.max(y_vals , axis=0)  # (d,)
        
        length = self.maxs - self.mins
        # Increase the boundary by 10% of the length
        self.mins = self.mins - 0.1 * length
        self.maxs = self.maxs + 0.1 * length
        
        self.outer_hull = self.get_data_boundary_box()
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

    # @torch.no_grad()
    def soft_quantize(self, x, temp=0.1):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        cdist_list = torch.cdist(x, self.protos, p=2)
        soft_pos = F.softmin(cdist_list/temp, dim=1)
        return soft_pos  # Return the index of nearest prototype


    def get_data_boundary_box(self):
        mins, maxs = self.mins, self.maxs
        if len(mins) == 1:
            # 1D case: Return the boundary as a tuple (min, max)
            return (mins[0], maxs[0])
        elif len(mins) == 2:
            # 2D case: Use ConvexHull to get the boundary
            bounds = [(mins[i], maxs[i]) for i in range(len(mins))]
            # Generate all possible combinations of mins and maxs (2^d corner points)
            corner_points = np.array(list(product(*bounds)))
            return ConvexHull(corner_points)
        else:
            raise ValueError("Unsupported dimensionality for get_data_boundary_box")
        
    # Return area of each proto
    def get_areas_pyvoro(self):
        if self.dims == 1:
            warnings.warn("Pyvoro is not implemented for 1d. Running Sklearn Voronoi!")
            return self.get_areas_voronoi()
        elif self.dims in (2,3):
            self.compute_pyvoro_voroni()
            return np.array(self.volumes)
        else:
            raise ValueError("Unsupported dimensionality for get_areas")
    
    def get_adjancencies_and_volumes(self):
        if self.dims == 1:
            protos_np =  self.get_protos_numpy()
            sorted_indices = protos_np.argsort()
            adjacenst_cells_all = []
            for i, sorted_pos in enumerate(sorted_indices):
                if i==0:
                    adjacenst_cells_all.append([sorted_indices[i+1]])
                elif i==(len(sorted_indices)-1):
                    adjacenst_cells_all.append([sorted_indices[i-1]])
                else:
                    adjacenst_cells_all.append([sorted_indices[i-1],sorted_indices[i+1]])
            volumes = self.get_areas_voronoi()
        elif self.dims in (2,3):
            self.compute_pyvoro_voroni()
            adjacenst_cells_all = self.adjacenst_cells_all
            volumes = self.volumes
        else:
            raise NotImplementedError(f"Not implemented for dims {self.dims}")
        return (adjacenst_cells_all, volumes) 

    
    def get_areas_voronoi(self):
        if hasattr(self.outer_hull,"points"):
            outer_point_list = self.outer_hull.points[self.outer_hull.vertices]
        else:
            # outer_point_list = torch.tensor(self.outer_hull)
            outer_point_list = np.array(self.outer_hull)
        return get_voronoi_areas(self.protos.detach().cpu().numpy(), outer_point_list)

    # Return area of each proto
    def get_proto_decision_boundaries(self):
        if self.dims == 1:
            if hasattr(self.outer_hull,"points"):
                outer_point_list = self.outer_hull.points[self.outer_hull.vertices]
            else:
                # outer_point_list = torch.tensor(self.outer_hull)
                outer_point_list = np.array(self.outer_hull)
            return get_voronoi_boundaries(self.protos.detach().cpu().numpy(), outer_point_list)
        elif self.dims in (2,3):
            self.compute_pyvoro_voroni()
            return self.polygons_all
        else: 
            raise ValueError("Unsupported dimensionality for get_proto_decision_boundaries")

    def get_voronoi_diagram(self):
        return Voronoi(self.protos.detach().cpu().numpy())
    
    def compute_pyvoro_voroni(self):
        if self.dims == 1:
            pass
        if self.dims==2:
            cells = pyvoro.compute_2d_voronoi(self.get_protos_numpy(), [[self.mins[0], self.maxs[0]], [self.mins[1], self.maxs[1]]], 2.0) # 2nd argum is bbox , # 3rd is block size    

        if self.dims == 3:
            cells = pyvoro.compute_voronoi(self.get_protos_numpy(), [[self.mins[0], self.maxs[0]], [self.mins[1], self.maxs[1]], [self.mins[2], self.maxs[2]]], 2.0)

        self.polygons_all = []
        self.adjacenst_cells_all = []
        self.centers = []
        self.volumes = []

        for i, cell in enumerate(cells):    
            polygon = cell['vertices']
            # print(np.array(polygon))
            # plt.fill(*zip(*polygon), color = color_map[tuple(cell['original'])], alpha=0.5)
            adjacent_cells = []
            polygons = []
            for j, item in enumerate(cell['faces']):
                adjacent_cells.append(item['adjacent_cell'])
            
            self.polygons_all.append(polygon)
            self.centers.append(cell['original'])
                
            self.adjacenst_cells_all.append(adjacent_cells)  
            self.volumes.append(cell['volume'])
    
    @torch.no_grad()    
    def clamp_protos(self):
        mins = torch.from_numpy(self.mins).to(self.protos.device)  # Shape: [feature_dim]
        maxs = torch.from_numpy(self.maxs).to(self.protos.device)  # Shape: [feature_dim]
        self.protos.data.clamp_(min=mins, max=maxs)
        

    def get_protos_numpy(self):
        return self.protos.detach().cpu().numpy()
    
class VoronoiQuantizer(GridQuantizer):
    def __init__(self, y_vals, proto_count_per_dim):
        super(VoronoiQuantizer, self).__init__(y_vals, proto_count_per_dim)
        self.protos.requires_grad = True
        
    def quantize(self, x):
        return super(VoronoiQuantizer, self).quantize(x)

    # Delete the prototypes in the given indices
    @torch.no_grad()
    def remove_proto(self, indices):
        mask = torch.full((len(self.protos),), True, dtype=bool)
        mask[indices] = False
        new_protos = self.protos[mask]
        self.protos = nn.Parameter(new_protos, requires_grad=True)

    # Repeat the prototypes in the given indices and concat to the end
    @torch.no_grad()
    def add_proto(self, indices):
        mask = torch.full((len(self.protos),), False, dtype=bool)
        mask[indices] = True
        selected_protos = self.protos[mask]
        noisy_copies = selected_protos + (0.01**0.5)*torch.randn_like(selected_protos)

        new_protos = torch.vstack((self.protos, noisy_copies))
        self.protos = nn.Parameter(new_protos, requires_grad=True)
    
    
#%%
import numpy as np
from itertools import product

class QuadTreeNode:
    def __init__(self, center, length):
        """
        Initialize a quadtree node in d dimensions.
        
        Parameters:
        - center: A list or numpy array of the coordinates of the center of the node in d dimensions.
        - length: A list or numpy array representing the length of the node along each dimension.
        """
        self.center = np.array(center)  # Center of the node
        self.length = np.array(length)  # Length along each dimension
        self.is_leaf = True             # Indicates whether the node is a leaf or subdivided
        self.children = []              # Holds children when the node is subdivided

    def split(self):
        """Subdivide the node into 2^d smaller nodes."""
        d = len(self.center)  # Number of dimensions
        half_length = self.length / 2
        
        # Generate all possible combinations of half-length offsets (-half_length, +half_length) along each dimension
        offsets = list(product(*[[-hl, hl] for hl in half_length]))
        
        # Create child nodes at the new centers
        for offset in offsets:
            new_center = self.center + np.array(offset)
            self.children.append(QuadTreeNode(new_center, half_length))
        
        self.is_leaf = False

    def merge(self):
        """Merge the node back into a single node, removing all children."""
        self.children = []
        self.is_leaf = True

    def size(self):
        """Compute the size of the node (length, area, volume, etc.)."""
        return np.prod(self.length)

class QuadTree:
    def __init__(self, center, length, max_depth):
        """
        Initialize the quadtree for d dimensions.
        
        Parameters:
        - center: A list or numpy array of the center of the root node.
        - length: A list or numpy array of the length along each dimension.
        - max_depth: The maximum depth the tree is allowed to reach.
        """
        self.root = QuadTreeNode(center, length)
        self.max_depth = max_depth

    def split_node(self, node, current_depth):
        """Recursively split a node if the max depth hasn't been reached."""
        if current_depth < self.max_depth and node.is_leaf:
            node.split()
            for child in node.children:
                self.split_node(child, current_depth + 1)

    def merge_node(self, node):
        """Recursively merge a node and its children."""
        if not node.is_leaf:
            for child in node.children:
                self.merge_node(child)
            node.merge()

    def get_leaf_nodes(self, node=None):
        """Recursively retrieve all leaf nodes."""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return [node]
        else:
            leaf_nodes = []
            for child in node.children:
                leaf_nodes.extend(self.get_leaf_nodes(child))
            return leaf_nodes

    def print_leaf_nodes(self):
        """Print details about all leaf nodes."""
        leaf_nodes = self.get_leaf_nodes()
        for i, node in enumerate(leaf_nodes):
            print(f"Leaf {i}: Center = {node.center}, Lengths = {node.length}, Size = {node.size()}")

# Example usage
if __name__ == "__main__":
    # Create a 2D quadtree with a center at (0, 0) and length (1, 1) along each dimension
    center = [0, 0]  # 2D center
    length = [1, 1]  # 2D length (size of space along x and y)
    max_depth = 3     # Maximum depth of the tree

    # Initialize the quadtree
    qt = QuadTree(center, length, max_depth)

    # Split the root node recursively until max depth is reached
    qt.split_node(qt.root, 0)

    # Print all leaf nodes
    qt.print_leaf_nodes()

    # Merge all nodes back into a single node
    qt.merge_node(qt.root)

    print("After merging:")
    qt.print_leaf_nodes()
