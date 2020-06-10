import numba as nb

# Make array type.  Type-expression is not supported in jitclasses
float_array_type = nb.types.float64[:]


# every jitclass in Numba needs a spec list containing the type of every variable it contains.
empty_spec = [('id', nb.types.string),
              ('type', nb.types.string),
              ('history', nb.types.DictType(nb.types.unicode_type, float_array_type))
              ]


@nb.experimental.jitclass(empty_spec)
class Empty(object):
    """The most basic module. It does nothing but has an ID and a type.
    """
    def __init__(self):
        self.id = "Empty"
        self.type = "Generic"
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        pass
