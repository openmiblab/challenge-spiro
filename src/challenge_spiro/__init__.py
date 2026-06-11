# List all submissions here 

# The LS solution is included here so it can be used as benchmark
from .utils.solution_ls import inverse as inverse_ls
from .utils.solution_benchmark import inverse as inverse_const

SUBMISSIONS = {
    # 'Least Squares': inverse_ls,
    'Dummy': inverse_const,
}

