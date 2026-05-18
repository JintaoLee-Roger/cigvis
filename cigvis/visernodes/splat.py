import numpy as np

from .base import ViserNodeMixin, auto_scale_from_points


class GaussianSplatNode(ViserNodeMixin):

    def __init__(
        self,
        centers,
        covariances,
        rgbs,
        opacities,
        scale=-1,
    ):
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        covariances = np.ascontiguousarray(covariances, dtype=np.float32)
        rgbs = np.ascontiguousarray(rgbs, dtype=np.float32)
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)

        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")
        n = centers.shape[0]
        if covariances.shape != (n, 3, 3):
            raise ValueError("covariances must have shape (N, 3, 3)")
        if rgbs.shape != (n, 3):
            raise ValueError("rgbs must have shape (N, 3)")
        if opacities.shape != (n, 1):
            raise ValueError("opacities must have shape (N, 1)")

        if isinstance(scale, (int, float)) and scale < 0:
            scale = auto_scale_from_points(centers, target=1.0)

        self._base_centers = centers
        self._base_covariances = covariances
        self.rgbs = np.ascontiguousarray(
            np.clip(rgbs, 0.0, 1.0),
            dtype=np.float32,
        )
        self.opacities = np.ascontiguousarray(
            np.clip(opacities, 0.0, 1.0),
            dtype=np.float32,
        )
        self.centers = centers
        self.covariances = covariances
        self._init_node_state('gaussian-splats', scale)
        self._on_scale_changed()

    @property
    def data_extent(self):
        return np.ptp(self._base_centers[:, :3], axis=0)

    def _on_scale_changed(self):
        scale = np.asarray(self.scale, dtype=np.float32)
        self.centers = np.ascontiguousarray(
            self._base_centers * scale,
            dtype=np.float32,
        )
        self.covariances = (
            self._base_covariances
            * scale[None, :, None]
            * scale[None, None, :]
        )
        self.covariances = np.ascontiguousarray(
            self.covariances,
            dtype=np.float32,
        )

    def update_node(self):
        if self.server is None:
            return
        self.nodes = self.server.scene.add_gaussian_splats(
            self.name,
            self.centers,
            self.covariances,
            self.rgbs,
            self.opacities,
        )
