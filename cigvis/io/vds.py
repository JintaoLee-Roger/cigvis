# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import Dict, List, Tuple
import numpy as np
import cigvis


class VDSReader:
    """
    A vds file reader that mimics the numpy array style.

    Examples
    ---------
    >>> d = VDSReader('test.vds')
    >>> d.shape
    # (601, 203, 400) # (n-inline, n-xline, n-time)
    >>> k = d[20:100, 100:200, 100:300] # k is np.array
    >>> k = d[20, :, :] # shape is (203, 400)
    >>> k = d[20, ...] # same as k = d[20, :, :]
    >>> k = d[20] # same as k = d[20, :, :]
    >>> k = d[:, 20, :]
    >>> print(d.min(), d.max())
    # will call `Layout.getChannelDescriptor(0).getValueRangeMin()`
    # and `Layout.getChannelDescriptor(0).getValueRangeMax()`
    """

    def __init__(self, filename) -> None:
        try:
            import openvds
        except ImportError:
            error_message = (
                "Failed to import openvds. Please follow these instructions based on your operating system:\n"
                "- For Linux or Windows, use: pip3 install openvds\n"
                "- For macOS, download the appropriate .whl file from 'https://github.com/JintaoLee-Roger/open-vds' "
                "and install it using: pip3 install <filename>.whl")
            raise ImportError(error_message)

        self.vds = openvds.open(filename)
        self.access_manager = openvds.getAccessManager(self.vds)
        self.layout = openvds.getLayout(self.vds)
        self.ch_des = self.layout.getChannelDescriptor(0)
        # (ni, nx, nt)
        self._shape = (self.layout.getDimensionNumSamples(2),
                       self.layout.getDimensionNumSamples(1),
                       self.layout.getDimensionNumSamples(0))
        self._close = openvds.close

    def read(self, ib, ie, xb, xe, tb, te) -> np.ndarray:
        """
        read sub volume
        """
        req = self.access_manager.requestVolumeSubset(min=(tb, xb, ib),
                                                      max=(te, xe, ie))
        shape = [ie - ib, xe - xb, te - tb]
        while shape.count(1) > 0:
            shape.remove(1)
        shape = tuple(shape)
        if len(shape) == 0:
            return req.data[0]
        return req.data.reshape(shape)

    @property
    def shape(self) -> Tuple:
        if not cigvis.is_line_first():
            return self._shape[::-1]
        return self._shape

    def __getitem__(self, slices) -> np.ndarray:
        idx = self._process_keys(slices)
        # to seismic index
        if not cigvis.is_line_first():
            idx = [idx[4], idx[5], idx[2], idx[3], idx[0], idx[1]]

        self._check_bound(*idx)
        data = self.read(*idx)

        if cigvis.is_line_first():
            return data  # seismic index
        else:  # seismic transpose index
            return data.transpose()

    def _process_keys(self, key) -> List:
        if isinstance(key, (int, np.integer)):
            if key < 0:
                key += self.shape[0]
            if key < 0 or key >= self.shape[0]:
                raise IndexError("Index out of range")
            return key, key + 1, 0, self.shape[1], 0, self.shape[2]
        elif key is Ellipsis:
            return 0, self.shape[0], 0, self.shape[1], 0, self.shape[2]
        elif isinstance(key, Tuple):
            num_ellipsis = key.count(Ellipsis)
            if num_ellipsis > 1:
                raise ValueError("Only one ellipsis (...) allowed")
            elif num_ellipsis == 1:
                key = (key[0], slice(None, None,
                                     None), slice(None, None, None))

            start_idx = [None] * 3
            end_idx = [None] * 3
            for i, k in enumerate(key):
                if k is None:
                    continue
                if isinstance(k, (int, np.integer)):
                    if k < 0:
                        k += self.shape[i]
                    start_idx[i] = k
                    end_idx[i] = k + 1
                elif isinstance(k, slice):
                    if not (k.step is None or k.step == 1):
                        raise IndexError(
                            f"only support step is 1, while got a step {k.step} in the {i}th dimension"
                        )

                    start_idx[i] = k.start or 0
                    end_idx[i] = k.stop or self.shape[i]

            ib, xb, tb = start_idx
            ie, xe, te = end_idx

            return ib, ie, xb, xe, tb, te
        else:
            raise IndexError("Invalid index slices")

    def _check_bound(self, ib, ie, xb, xe, tb, te) -> None:
        assert ib < ie and ie <= self._shape[0]
        assert xb < xe and xe <= self._shape[1]
        assert tb < te and te <= self._shape[2]

    def max(self) -> float:
        return self.ch_des.getValueRangeMax()

    def min(self) -> float:
        return self.ch_des.getValueRangeMin()

    def close(self) -> None:
        self._close(self.vds)

    def close(self) -> None:
        self.segy.close_file()

    def __array__(self):
        """To support np.array(SegyNP(xxx))"""
        return self[...]
        
    def to_numpy(self):
        """like pandas"""
        return self[...]

    def __array_function__(self, func, types, args, kwargs):
        if func is np.nanmin:
            return self.min()
        elif func is np.nanmax:
            return self.max()
        raise NotImplementedError(f"Function {func} is not implemented for SegyNP")


def create_vds_from_array(d: np.ndarray,
                          vdsf: str,
                          clim: List = None,
                          brick_size: int = 64):
    """
    create a vds file from a np.ndarray,
    check the official repository to view the original examples,
    path: `open-vds/examples/NpzToVds/npz_to_vds.py`,
    repository: 
    https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-vds

    Parameters
    -----------
    d : array-like
        input data
    vdsf : str
        output vdsfile
    clim : List
        [vmin, vmax], if None, will use [d.min(), d.max()]
    brick_size : int
        brick size, can be one of {32, 64, 128, 256, 512, 1024, 2048}
    """
    try:
        import openvds
    except ImportError:
        error_message = (
            "Failed to import openvds. Please follow these instructions based on your operating system:\n"
            "- For Linux or Windows, use: pip3 install openvds\n"
            "- For macOS, download the appropriate .whl file from 'https://github.com/JintaoLee-Roger/open-vds' "
            "and install it using: pip3 install <filename>.whl")
        raise ImportError(error_message)

    if cigvis.is_line_first():
        ni, nx, nt = d.shape
    else:
        nt, nx, ni = d.shape

    brick_size_swith = {
        32: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_32,
        64: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_64,
        128: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_128,
        256: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_256,
        512: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_512,
        1024: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_1024,
        2048: openvds.VolumeDataLayoutDescriptor.BrickSize.BrickSize_2048
    }
    brick_size = brick_size_swith[brick_size]

    if clim is None:
        clim = [d.min(), d.max()]

    layout_des = openvds.VolumeDataLayoutDescriptor(
        brick_size,
        0,
        0,
        4,
        openvds.VolumeDataLayoutDescriptor.LODLevels.LODLevels_None,
        openvds.VolumeDataLayoutDescriptor.Options.Options_None,
    )

    axis_des = [
        openvds.VolumeDataAxisDescriptor(nt, 'Sample', 'ms', 0.0, nt * 2),
        openvds.VolumeDataAxisDescriptor(nx, 'Crossline', 'm', 0.0, nx),
        openvds.VolumeDataAxisDescriptor(ni, 'Inline', 'm', 0.0, ni)
    ]

    channel_des = [
        openvds.VolumeDataChannelDescriptor(
            openvds.VolumeDataChannelDescriptor.Format.Format_R32,
            openvds.VolumeDataChannelDescriptor.Components.Components_1,
            "Value",
            'unitless',
            clim[0],
            clim[1],
            flags=openvds.VolumeDataChannelDescriptor.Flags.NoLossyCompression)
    ]

    metaData = openvds.MetadataContainer()
    metaData.setMetadataDoubleVector2(
        openvds.KnownMetadata.surveyCoordinateSystemOrigin().category,
        openvds.KnownMetadata.surveyCoordinateSystemOrigin().name,
        (1234.0, 4321.0))
    vds = openvds.create(vdsf, layout_des, axis_des, channel_des, metaData)

    layout = openvds.getLayout(vds)
    manager = openvds.getAccessManager(vds)
    accessor = manager.createVolumeDataPageAccessor(
        openvds.DimensionsND.Dimensions_012, 0, 0, 8,
        openvds.IVolumeDataAccessManager.AccessMode.AccessMode_Create, 1024)

    rang = range
    try:
        from tqdm import trange
        rang = trange
    except:
        pass

    for c in rang(accessor.getChunkCount()):
        page = accessor.createPage(c)
        buf = np.array(page.getWritableBuffer(), copy=False)
        (rmin, rmax) = page.getMinMax()
        if cigvis.is_line_first():
            buf[:, :, :] = d[rmin[2]:rmax[2], rmin[1]:rmax[1], rmin[0]:rmax[0]]
        else:
            buf[:, :, :] = d[rmin[0]:rmax[0], rmin[1]:rmax[1],
                             rmin[2]:rmax[2]].T

        page.release()

    accessor.commit()
    openvds.close(vds)
