from challenge_spiro.utils.model import forward_model, forward
from challenge_spiro.utils.dro import generate_subject
from challenge_spiro.utils.solution_ls import inverse as inverse_ls
from challenge_spiro.utils.solution_benchmark import inverse as inverse_const


truth, time, signal = generate_subject()

aorta_liver = forward_model(*truth, time)
# aorta_liver.plot(time, signal)

recon = inverse_ls(time, signal)

print( [100 * (recon[i] - truth[i]) / truth[i] for i in range(len(truth))])

signal_recon = forward(*recon, time)
aorta_liver.plot(time, signal_recon)