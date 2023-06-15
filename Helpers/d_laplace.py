
from ldp_base import LDPBase


class NDLaplaceMechanismBase:
    def randomise(self, value):
        raise NotImplementedError

class NDLaplaceMechanism(LDPBase):
    def __init__(self, epsilon, domain):
        self._domain = domain
        self._epsilon = epsilon
        self._pm_encoder = NDLaplaceMechanismBase(epsilon=epsilon)

    def _transform(self, value):
        """transform v in self.domain to v' in [-1,1]"""
        value = self._check_value(value)
        a, b = self._domain
        return (2*value - b - a) / (b - a)

    def _transform_T(self, value):
        """inverse of self._transform"""
        a, b = self._domain
        return (value * (b-a)+a+b)/2

    def randomise(self, value):
        value = self._transform(value)
        value = self._pm_encoder.randomise(value)
        value = self._transform_T(value)
        return value

    def _check_value(self, value):
        if not self._domain[0] <= value <= self._domain[1]:
            raise ValueError("the input value={} is not in domain={}".format(value, self._domain))
        return value