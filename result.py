import os

import numpy as np


def crowdflow_result(root, modes, acts, penalties):
    crowdflow_root = os.path.join(root, "CrossVal")
    crosses = ["B", "C", "D", "E", "A"]
    c2s = {"A": "5", "B": "1", "C": "2", "D": "3", "E": "4"}

    print("MAE\tMAE-std\tRMSE\tRMSE-std\tpix-MAE\tpix-MAE-std\tpix-RMSE\tpix-RMSE-std")
    for m in modes:
        for c in crosses:
            for act in acts:
                for penalty in penalties:
                    res_path = os.path.join(crowdflow_root, 
                                            c, 
                                            "{}_{}_{}".format(m, act, penalty), 
                                            "baseline", 
                                            c2s[c], 
                                            "result.npz")
                    # print(c2s[c], m, act, penalty)
                    res = np.load(res_path)
                    print("{} {} {} {} {} {} {} {}".format(
                          res["mae"], res["mae_std"], 
                          res["rmse"], res["rmse_std"], 
                          res["pix_mae_val"], res["pix_mae_val_std"],
                          res["pix_rmse_val"], res["pix_rmse_val_std"]))

def main():
    root = "/groups1/gca50095/aca10350zi/habara_exp"

    modes = ["once", "add"]
    acts = ["relu"]
    penalties = ["0.0", "0.1", "0.01"]

    crowdflow_result(root, modes, acts, penalties)

if __name__ == "__main__":
    main()
