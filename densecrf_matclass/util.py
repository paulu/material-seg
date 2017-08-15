import os
import json
import numpy as np


def hex_to_rgb(hex_str):
    colorstring = hex_str.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]
    if len(colorstring) != 6:
        raise ValueError("input #%s is not in #RRGGBB format" % colorstring)
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)


def roundint(x):
    return int(round(x))


def mkdir_p(outdir):
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except IOError as e:
            print e


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
