import numpy as np
from magicgui import magic_factory
from ._points_util import slice_points
from profilehooks import profile


@magic_factory(
        call_button='Merge',
        viewer={'visible': False},
        ndim={'min': 2, 'max': 3},
        )
def merge_labels(
        viewer: 'napari.viewer.Viewer',
        labels: 'napari.layers.Labels',
        points: 'napari.layers.Points',
        ndim: int = 3,
        ):

    """Merge labels based on provided points.

    Only labels containing points will be modified. The operation will be
    performed in-place on the input Labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The current viewer displaying the data.
    labels : napari Labels layer
        The layer containing the segmentation. This will be used both for
        input and output.
    points : napari Points layer
        The points marking the labels to be split.
    ndim : int in {2, 3}
        The number of dimensions for the watershed operation.
    """
    if len(points.data) == 0:
        return
    coord = viewer.dims.current_step
    slice_idx = coord[:-ndim]
    # find the labels corresponding to the current points in the points layer
    labels_sliced = np.asarray(labels.data[slice_idx])
    points_sliced = slice_points(points, viewer.dims, ndim)
    points_data_to_world = points._transforms[1:3].simplified
    labels_world_to_data = labels._transforms[1:3].simplified.inverse
    points_transformed = labels_world_to_data(
            points_data_to_world(points_sliced)
            ).astype(int)[:, -ndim:]

    future = labels.data[slice_idx].write(_merge_labels(labels_sliced, points_transformed))
    future.add_done_callback(lambda _: labels.refresh())
    points.data = np.empty((0, viewer.dims.ndim), dtype=float)


def _merge_labels(labels, points):
    points = np.round(points).astype(int)
    coords = tuple([points[:, i] for i in range(points.shape[1])])
    p_lab = labels[coords]
    p_lab = np.unique(p_lab)
    p_lab = p_lab[p_lab != 0]
    min_selected_lab = p_lab.min()
    for lab in p_lab:
        labels[lab == labels] = min_selected_lab
    return labels

