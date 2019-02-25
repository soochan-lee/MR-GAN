from collections import namedtuple
from torch import nn
import contextlib
import time
import itertools
import yaml


# =============
# PyTorch Utils
# =============

class Lambda(nn.Module):
    def __init__(self, f=None):
        super().__init__()
        self.f = f if f is not None else (lambda x: x)

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


# ============
# Config Utils
# ============

def load_config(path,  as_namedtuple=True):
    config = yaml.load(open(path)) or {}
    return dict_to_namedtuple(config) if as_namedtuple else config


def dict_to_namedtuple(d):
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namedtuple(v)
        return namedtuple('d', d.keys())(**d)
    return d


def namedtuple_to_dict(n):
    def _isnamedtuple(x):
        cls = type(x)
        bases = cls.__bases__
        fields = getattr(cls, '_fields', None)

        if len(bases) != 1 or bases[0] != tuple:
            return False

        if not isinstance(fields, tuple):
            return False

        return all(type(name) == str for name in fields)

    d = dict(n._asdict())
    for k, v in d.items():
        if _isnamedtuple(v):
            d[k] = namedtuple_to_dict(v)
    return d


# ============
# Python Utils
# ============

@contextlib.contextmanager
def time_logging_context(start_message, ending_message='done'):
    start = time.clock()
    print(start_message, end=' ', flush=True)
    yield
    took = time.clock() - start
    print(ending_message + ' (took {:.03f} seconds)'.format(took))


def ncycle(iterable, n):
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def updated_nt(nt, path, value):
    try:
        attr_name, attr_names_left = path.split('.', 1)
        attr = getattr(nt, attr_name)
        return _updated_nt(nt, attr_name, updated_nt(
            attr, attr_names_left, value
        ))
    except ValueError:
        return _updated_nt(nt, path, value)


def _updated_nt(nt, name, value):
    mapping = nt._asdict()
    mapping[name] = value
    return dict_to_namedtuple(mapping)
