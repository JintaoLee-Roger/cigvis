
# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import Dict, List, Tuple
import numpy as np

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