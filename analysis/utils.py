import wandb
import numpy as np
from collections import defaultdict


def load_wandb(project):
    api = wandb.Api()
    runs = api.runs(project)

    info = defaultdict(dict)

    for i in runs:
        if i.name.split("_")[0] == "decoder" or i.name.split("_")[0] == "encoder":
            print("run name = ", i.name, " id: ", i.id)
            run_info = defaultdict(dict)
            run_info["train"]["loss"] = np.array(
                i.history(keys=["train/PackedNLLLoss"], samples=1000)
            )
            for metric in [
                "MAE_mean",
                "MSE_mean",
                "MAE_median",
                "MSE_median",
                "ND",
                "CRPS",
                "RMSE",
                "MASE",
                "MAPE",
                "PackedNLLLoss",
                "SMAPE",
                "NRMSE",
            ]:
                run_info["monash"][metric] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_0"], samples=100)
                )
                run_info["lsf"][metric] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_1"], samples=100)
                )
                run_info["val"][metric] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_2"], samples=100)
                )

            info[i.name] = run_info

    return info


def load_wandb_repeat(path):
    
    api = wandb.Api()
    runs = api.runs(path)

    info = defaultdict(dict)
    for i in runs:
        print("run name = ", i.name, " id: ", i.id)
        
        run_info = {
            "train": defaultdict(dict),
            "monash": defaultdict(dict),
            "lsf": defaultdict(dict),
            "val": defaultdict(dict),
        }
        
        if i.name.split("_")[0] == "encoder":
            if len(i.name.split("_")) == 2:
                index_number = 1
            else:
                index_number = int(i.name.split("_")[2][-1])

            run_info["train"]["loss"][index_number] = np.array(
                i.history(keys=["train/PackedNLLLoss"], samples=1000)
            )
            for metric in [
                "MAE_mean",
                "MSE_mean",
                "MAE_median",
                "MSE_median",
                "CRPS",
                "ND",
                "RMSE",
                "MASE",
                "MAPE",
                "PackedNLLLoss",
                "SMAPE",
                "NRMSE",
            ]:
                run_info["monash"][metric][index_number] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_0"], samples=100)
                )
                run_info["lsf"][metric][index_number] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_1"], samples=100)
                )
                run_info["val"][metric][index_number] = np.array(
                    i.history(keys=["val/" + metric + "/dataloader_idx_2"], samples=100)
                )

        info[i.name] = run_info
    return info


MODELSIZE2PARAM = {
    '1K': 872,
    '10K': 7032,
    '100K': 100928,
    '300K': 302656,
    '1M': 1771712,
    '3M': 4747264,
    '10M': 1.41 * 1e7,
    '30M': 3.78 * 1e7,
    '100M': 1.27 * 1e8,
    '300M': 3.02 * 1e8,
    '1B': 1.016 * 1e9,
}


DATASIZE2NUM = {
    '10M': 1*1e7,
    '100M': 1*1e8,
    '1B': 1.03*1e9,
    '5B': 4.8 * 1e9,
    '16B': 15.7 * 1e9,
}

def fit_powerlaw(x, y, signal='A'):
    logx = np.log10(x)
    logy = np.log10(y)
    
    coeffs = np.polyfit(logx, logy, 1)
    poly = np.poly1d(coeffs)
    logy_fit = poly(logx)
    
    k, b = coeffs
    x0 = 10**(-b/k)
    x0_sci = f'{x0:.1e}'
    mantissa, exponent = x0_sci.split('e')
    mantissa = float(mantissa)
    exponent = int(exponent)
    label = r'$L(%s) = \left({%.1f} \cdot 10^{%d} / %s  \right)^{%.3f}$' % (signal, mantissa, exponent, signal, -k)
    
    return logy_fit, label


def step_to_compute(size):
    B = 128
    N = MODELSIZE2PARAM[size]
    L = 512
    step = np.arange(1, 101)
    C = 6 * B * N * L
    return C * step / (8.64*1e19)

