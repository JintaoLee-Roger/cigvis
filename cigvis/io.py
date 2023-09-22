# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
geophysics data io, 

Including:
    - well logs (.las format)
    - vds file (need install openvds)
    
"""

from typing import Dict, List, Tuple
import numpy as np
import cigvis


def load_las(fpath: str) -> Dict:
    """
    Load las format file (Well Logs).
    Return a dict contains all informations of the las file.

    Each section is a dict, which contains `name`, 'unit', 'value' 
    and 'descreption'
    
    Use `data = las['data']` to obtain data, `las['CURVE']['name']`
    to obtain data name of each column
    
    Parameters
    ----------
    fpath : str
        las file path
    
    Returns
    -------
    las : Dict
        las data and its information

    Examples
    ----------
    >>> las = load_las('log01.las')
    >>> las.keys() # sections
    dict_keys(['VERSION', 'WELL', 'CURVE', 'PARAMETER', 'data'])
    >>> data = las['data']
    >>> data.shape
    (921, 5)
    >>> curves = las['CURVE'] # curves
    >>> curves['name']
    ['DEPT', 'ILD', 'DPHI', 'NPHI', 'GR']
    >>> curves['unit']
    ['M', 'OHMM', 'V/V', 'V/V', 'API']
    >>> las['WELL']['name'] # meta information
    ['WELL', 'LOC', 'UWI', 'ENTR', 'SRVC', 'DATE', 'STRT', 'STOP', 'STEP', 'NULL']
    >>> las['WELL]['value][1] # location
    '00/01-12-079-14W4/0'
    """

    with open(fpath, 'r') as f:
        text = f.read()

    sections = [s.strip() for s in text.split('~') if s.strip() != '']
    las = {}
    for section in sections:
        # data
        if section[:2] == 'A ' or section[:5] == 'Ascii':
            section = [
                l.strip() for l in section.split('\n')
                if l.strip() != '' and l.strip()[0] != '#'
            ]
            data = np.loadtxt(section[1:], np.float32)
            las['data'] = data
        else:  # sections
            name, cdict = _process_las_section(section)
            las[name] = cdict

    return las


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
    import openvds
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
        import openvds

        self.vds = openvds.open(filename)
        self.access_manager = openvds.getAccessManager(self.vds)
        self.layout = openvds.getLayout(self.vds)
        self.ch_des = self.layout.getChannelDescriptor(0)
        # (ni, nx, nt)
        self._shape = (self.layout.getDimensionNumSamples(2),
                       self.layout.getDimensionNumSamples(1),
                       self.layout.getDimensionNumSamples(0))

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
        if isinstance(key, int):
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
                if isinstance(k, int):
                    if k < 0:
                        k += self.shape[i]
                    start_idx[i] = k
                    end_idx[i] = k + 1
                elif isinstance(k, slice):
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


##### private functions, don't call them ##############


def _process_las_line(line: str) -> Tuple:
    parts = [l.strip() for l in line.strip().split(':')]
    descreption = parts[-1]
    line = ':'.join(parts[:-1]).strip()
    parts = [l for l in line.split('.')]
    name = parts[0].strip()
    line = '.'.join(parts[1:])

    if len(line) > 0 and line[0] != ' ':
        parts = line.split(' ')
        unit = parts[0].strip()
        value = ' '.join(parts[1:]).strip()
    else:
        unit = None
        value = line.strip()

    return name, unit, value, descreption


def _process_las_section(text: str) -> Tuple:
    text = [
        l.strip() for l in text.split('\n')
        if l.strip() != '' and l.strip()[0] != '#'
    ]
    name = text[0].split(' ')[0]
    cdict = {'name': [], 'unit': [], 'value': [], 'description': []}
    for t in text[1:]:
        out = _process_las_line(t)
        cdict['name'].append(out[0])
        cdict['unit'].append(out[1])
        cdict['value'].append(out[2])
        cdict['description'].append(out[3])
    return name, cdict
