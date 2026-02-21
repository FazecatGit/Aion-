from functools import reduce

def pipe(value, *fns):
    """Pipe value through sequence of functions"""
    return reduce(lambda x, f: f(x), fns, value)

def compose(*fns):
    """Compose functions left-to-right: compose(f, g)(x) = g(f(x))"""
    return reduce(lambda f, g: lambda x: g(f(x)), fns, lambda x: x)
