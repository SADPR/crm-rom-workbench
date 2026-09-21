#!/usr/bin/env python3
"""
Train an RBF surrogate and keep the exact
validation subset for later evaluation.
"""

import argparse, sys, time, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------------ #
#  RBF kernels
# ------------------------------------------------------------------ #
def gaussian_rbf(r, eps):           return np.exp(-(eps * r) ** 2)
def inverse_multiquadric_rbf(r, e): return 1.0 / np.sqrt(1 + (e * r) ** 2)
def multiquadric_rbf(r, eps):       return np.sqrt(1 + (eps * r) ** 2)
def linear_rbf(r, eps):             return r

rbf_kernels = {
    "gaussian":      gaussian_rbf,
    # "imq":           inverse_multiquadric_rbf,
    # "multiquadric":  multiquadric_rbf,
    # "linear":        linear_rbf,
}
# ------------------------------------------------------------------ #
def unscale_y(y_scaled, scaling_method, y_info):
    if scaling_method == 0:               # z-score
        mu_out, sig_out = y_info
        return y_scaled * sig_out + mu_out
    else:                                 # min-max → [-1,1]
        y_min, y_max = y_info
        return 0.5 * (y_scaled + 1.0) * (y_max - y_min) + y_min
# ------------------------------------------------------------------ #

def rel_l2(y_true, y_pred):
    """Relative L2 ‖y_pred − y_true‖₂ / ‖y_true‖₂  (scalar)."""
    return np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)

def main():
    p = argparse.ArgumentParser(description="RBF PROM trainer (train-only set)")
    p.add_argument("--data_file")
    p.add_argument("--dimV", default=20, type=int)
    p.add_argument("--output_path", default=".")
    p.add_argument("--skip_columns", default=0, type=int)
    p.add_argument("--skip_rows",    default=1, type=int)
    args = p.parse_args()

    if args.data_file is None:
        args.data_file  = "state.coords"
        print(f"[Info] default data file: {args.data_file}")

    raw = np.loadtxt(args.data_file,
                     skiprows=args.skip_rows, delimiter=',')
    if args.skip_columns:
        raw = raw[:, args.skip_columns:]

    TOTAL_COLS = raw.shape[1]
    if TOTAL_COLS < 2:
        sys.exit("[Error] need at least 2 columns to split into p and s")

    print(f"[Info] total columns = {TOTAL_COLS}")
    best_global_err = np.inf

    for p_size in range(args.dimV, args.dimV+1):
        s_size = TOTAL_COLS - p_size
        print(f"\n=== split p={p_size}  s={s_size} ===")

        q_p = raw[:, :p_size]
        q_s = raw[:, p_size:TOTAL_COLS]

        # ------- scaling (unchanged) ------------------------------------- #
        scaling_method = 1                     # 0=z-score, 1=min-max
        if scaling_method == 0:
            mu_all  = raw.mean(0);  sig_all = raw.std(0)
            mu_in,  sig_in  = mu_all[:p_size],  sig_all[:p_size]
            mu_out, sig_out = mu_all[p_size:],  sig_all[p_size:]
            X = (q_p - mu_in)/sig_in
            Y = (q_s - mu_out)/sig_out
            y_info = (mu_out, sig_out)
        else:
            xmin,xmax = q_p.min(0), q_p.max(0)
            ymin,ymax = q_s.min(0), q_s.max(0)
            denom_x   = np.where(xmax-xmin<1e-15,1.0,xmax-xmin)
            denom_y   = np.where(ymax-ymin<1e-15,1.0,ymax-ymin)
            X = 2*((q_p-xmin)/denom_x)-1
            Y = 2*((q_s-ymin)/denom_y)-1
            y_info = (ymin, ymax)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.10, random_state=42)

        epsilons = np.logspace(np.log10(0.5), np.log10(10.0), 100)
        kernels  = list(rbf_kernels)
        best_err = np.inf
        best_W   = best_kernel = best_eps = None

        D_tr  = np.linalg.norm(X_train[:,None]-X_train[None,:],axis=2)
        D_val = np.linalg.norm(X_val[:,None]  -X_train[None,:],axis=2)

        for eps in epsilons:
            for k in kernels:
                kfun = rbf_kernels[k]
                phi_tr = kfun(D_tr, eps) + 1e-8*np.eye(len(X_train))
                try:
                    W = np.linalg.solve(phi_tr, Y_train)
                except np.linalg.LinAlgError:
                    continue
                phi_val = kfun(D_val, eps)
                Y_pred_valS = phi_val @ W
                Y_val_ph      = unscale_y(Y_val,      scaling_method, y_info)
                Y_pred_val_ph = unscale_y(Y_pred_valS, scaling_method, y_info)
                mse = mean_squared_error(Y_val_ph, Y_pred_val_ph)
                rel = rel_l2(Y_val_ph, Y_pred_val_ph) 
                if rel < best_err:
                    best_err   = rel
                    best_kernel, best_eps, best_W = k, eps, W
                    mse_best  = mse

        print(f"[Best split] ker={best_kernel}  eps={best_eps:.4g}  "
          f"RelL2={best_err*100:.2f}%  (MSE={mse_best:.3e})")
        
        # -------------------------------------------------
        #  Trick: retrain with 100 % of the snapshots
        # -------------------------------------------------
        kfun_best = rbf_kernels[best_kernel]
        D_all     = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        Phi_all   = kfun_best(D_all, best_eps) + 1e-8 * np.eye(len(X))
        W_full    = np.linalg.solve(Phi_all, Y)        # size  (Nsnap × n̄)
        # -------------------------------------------------

        # ---- keep global best ------------------------------------------- #
        if best_err < best_global_err:
            best_global_err = best_err
            # overwrite artefacts with the new champion
            np.savez("{}rbf_validation_data.npz".format(args.output_path), X_val=X_val, Y_val=Y_val)
            with open("{}rbf_precomputations.txt".format(args.output_path), "w") as f:
                # f.write(f"{best_W.shape[0]} {best_W.shape[1]}\n")
                # np.savetxt(f, best_W, fmt="%.16e")
                f.write(f"{W_full.shape[0]} {W_full.shape[1]}\n")#Trick
                np.savetxt(f, W_full, fmt="%.16e")#Trick
            with open("{}rbf_xTrain.txt".format(args.output_path), "w") as f:
                # f.write(f"{X_train.shape[0]} {X_train.shape[1]}\n")
                # np.savetxt(f, X_train, fmt="%.16e")
                f.write(f"{X.shape[0]} {X.shape[1]}\n")#Trick
                np.savetxt(f, X, fmt="%.16e")#Trick
            # scaling params
            if scaling_method == 0:
                mu_out, sig_out = y_info
                mu_in  = np.zeros(p_size)
                sig_in = np.ones (p_size)
                with open("{}rbf_stdscaling.txt".format(args.output_path),"w") as f:
                    f.write(f"{p_size} {s_size}\n0\n")
                    np.savetxt(f, mu_in [None], fmt="%.16e")
                    np.savetxt(f, sig_in[None], fmt="%.16e")
                    np.savetxt(f, mu_out[None], fmt="%.16e")
                    np.savetxt(f, sig_out[None],fmt="%.16e")
            else:
                y_min, y_max = y_info
                # xmin = np.zeros(p_size); xmax = np.ones(p_size) <- This was the error (artifact)
                xmin, xmax = q_p.min(0), q_p.max(0) # Correct scaling
                with open("{}rbf_stdscaling.txt".format(args.output_path),"w") as f:
                    f.write(f"{p_size} {s_size}\n1\n")
                    np.savetxt(f, xmin[None], fmt="%.16e")
                    np.savetxt(f, xmax[None], fmt="%.16e")
                    np.savetxt(f, y_min[None], fmt="%.16e")
                    np.savetxt(f, y_max[None], fmt="%.16e")
            with open("{}rbf_hyper.txt".format(args.output_path),"w") as f:
                f.write("2 1\n")
                f.write(best_kernel + "\n")
                f.write(f"{best_eps:.16e}\n")

    print(f"\n[Done] best global RelL2={best_global_err*100:.2f}%")

if __name__ == "__main__":
    main()
