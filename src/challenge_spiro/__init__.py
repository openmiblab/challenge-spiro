
# try:    
#     # Private: Only challenge organisers have access
#     from .solutions.submitted._registered import REGISTERED
# except:
#     REGISTERED = {}
from .solutions.submitted._registered import REGISTERED

from .solutions.provided.benchmark import inverse as inverse_bench
from .solutions.provided.benchmark import inverse as inverse_norm




SOLUTIONS = REGISTERED | {
    'Normative': inverse_norm,
    'Benchmark': inverse_bench,
}

