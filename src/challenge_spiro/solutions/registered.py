
try:    
    # Private: Only challenge organisers have access
    from .submitted._registered import REGISTERED
except:
    REGISTERED = {}
# from .solutions.submitted._registered import REGISTERED

from .provided.benchmark import inverse as inverse_bench
from .provided.normative import inverse as inverse_norm




SOLUTIONS = REGISTERED | {
    'Normative': inverse_norm,
    'Benchmark': inverse_bench,
}

