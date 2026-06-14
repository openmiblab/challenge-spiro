# List all solutions here 

from .solutions.benchmark import inverse as inverse_bench
from .solutions.normative import inverse as inverse_norm

SOLUTIONS = {
    'Normative': inverse_norm,
    'Benchmark': inverse_bench,
}

