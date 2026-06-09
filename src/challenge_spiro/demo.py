from challenge_spiro.utils.model import forward_model, forward
from challenge_spiro.utils.ls import inverse_ls



    
truth = 0.001, 0.002, 0.0006, 0.0009

time, signal = forward(*truth)

# forward_model(*truth).plot(time, signal)

recon = inverse_ls(time, signal)

print(recon)

print( [100 * (recon[i] - truth[i]) / truth[i] for i in range(len(truth))])