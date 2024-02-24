import rasterio
import rasterio.features
from rasterio.transform import Affine
import numpy as np
import geojson
from typing import Dict


def labeled_image_to_geojson(lab: np.ndarray, *,
                             object_type='annotation',
                             connectivity: int = 4,
                             transform: Affine = None,
                             include_labels: bool = True,
                             label_map: Dict = None,
                             feat_collection: bool = False):
    """
    Args:
        lab: input labeled image
        object_type: objectType
        connectivity: Pixel connectivity
        transform: affine transformation. None --> no transformation
        include_labels: whether to include labels in props
        label_map: map the label value to name of "classification"
        feat_collection: return FeatureCollection or a list of features
    Returns:

    """
    feature_list = []
    lab = lab.astype(np.uint8)
    mask = lab > 0

    if transform is None:
        transform = Affine.scale(1.0)
    # Trace geometries
    for geom, value in rasterio.features.shapes(lab, mask=mask,
                                                connectivity=connectivity, transform=transform):

        # Create properties
        props = dict(objectType=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'Label', 'value': int(value)}]

        # Just to show how a classification can be added
        if label_map is not None:
            props['classification'] = label_map[int(value)]

        # Wrap in a dict to effectively create a GeoJSON Feature
        feature = geojson.Feature(geometry=geom, properties=props)

        feature_list.append(feature)
    if feat_collection:
        feature_list = geojson.FeatureCollection(feature_list)
    return feature_list
