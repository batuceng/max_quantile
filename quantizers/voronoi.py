import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import torch 
def get_voronoi_areas(points, boundaries):
    dims = points.shape[1]
    if dims==1:
        return boundary_clipped_voronoi_areas_1d(points, boundaries)
    elif dims==2:
        return boundary_clipped_voronoi_areas_2d(points, boundaries)
    else:
        raise NotImplementedError(f"get_voronoi_areas is not implemented for d:{dims}")


#!todo implement for other dimensions first for 1d

def boundary_clipped_voronoi_areas_2d(points, boundaries):
    vor = Voronoi(points)
    
    # Define the bounding box
    bounding_box = ShapelyPolygon(boundaries)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Store the areas of the clipped Voronoi regions
    clipped_areas = []

    # Iterate over each Voronoi region and calculate the clipped area
    for region in regions:
        polygon_points = vertices[region]
        voronoi_polygon = ShapelyPolygon(polygon_points)

        # Clip the Voronoi polygon by the bounding box
        clipped_polygon = voronoi_polygon.intersection(bounding_box)

        if not clipped_polygon.is_empty:
            # For MultiPolygon, sum the areas and plot each part
            if clipped_polygon.geom_type == 'Polygon':
                area = clipped_polygon.area
                clipped_areas.append(area)
            elif clipped_polygon.geom_type == 'MultiPolygon':
                area = 0
                for poly in clipped_polygon:
                    area += poly.area
                clipped_areas.append(area)
        else:
            clipped_areas.append(0)
    return clipped_areas


def voronoi_finite_polygons_2d(vor, radius=None):
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
        if radius is None:
            radius = np.ptp(vor.points).max() * 2
        center = vor.points.mean(axis=0)
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        new_regions = []
        new_vertices = vor.vertices.tolist()
        for p1, region_idx in enumerate(vor.point_region):
            vertices = vor.regions[region_idx]
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n

                far_point = vor.vertices[v2] + direction * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            vs = np.array([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        return new_regions, np.array(new_vertices)
    
    

def boundary_clipped_voronoi_areas_1d(points, boundaries):
    points, indices = torch.sort(points)
    a, b = boundaries
    n = len(points)
    device = points.device

    # Calculate midpoints between sorted points
    midpoints = (points[:-1] + points[1:]) / 2  # Shape: [n - 1]

    # Initialize left and right boundaries for each Voronoi cell
    left = torch.empty(n, device=device)
    right = torch.empty(n, device=device)

    # Set left boundaries
    left[0] = a
    left[1:] = midpoints  # Left boundaries for points 1 to n-1

    # Set right boundaries
    right[:-1] = midpoints  # Right boundaries for points 0 to n-2
    right[-1] = b

    # Clip left and right boundaries to the overall boundaries [a, b]
    left_clipped = torch.clamp(left, min=a)
    right_clipped = torch.clamp(right, max=b)

    # Calculate the lengths of the Voronoi cells
    lengths = torch.clamp(right_clipped - left_clipped, min=0)

    return lengths