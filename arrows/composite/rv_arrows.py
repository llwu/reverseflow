"""
This file contains random variable arrows with closed form quantile functions
for sampling. For any distribution with a quantile function Q, one can sample
from the distribution by taking Q(p), where p ~ U[0, 1].
"""

class ExponentialRV(CompositeArrow):
