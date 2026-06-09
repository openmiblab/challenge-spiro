from challenge_spiro.utils.model import forward_model, forward, inverse_ls, inverse_ls_model


if __name__=='__main__':
    
    ground_truth = 0.2, 0.001, 0.002, 0.0006, 0.0009
    experiment = {'TS': 2.5, 't_acq': (45, 45), 't_wait': 90}

    model = forward_model(*ground_truth, **experiment)
    time, signal = forward(*ground_truth, **experiment)

    model.plot(time, signal)