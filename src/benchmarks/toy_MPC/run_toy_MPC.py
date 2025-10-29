"""Run HCNN on toy MPC problem (PyTorch version, GPU ready)."""

import argparse
import datetime
import pathlib
import time
import timeit
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.toy_MPC.load_toy_MPC import ToyMPCDataset, load_data
from benchmarks.toy_MPC.model import setup_model
from benchmarks.toy_MPC.plotting import generate_trajectories, plot_training
from src.tools.utils import GracefulShutdown, Logger, load_configuration


# ============================================================
#   EVALUATION (PyTorch)
# ============================================================

def evaluate_hcnn(
    loader: ToyMPCDataset,
    model: nn.Module,
    batched_objective,
    A: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    prefix: str,
    device: torch.device,
    time_evals: int = 10,
    print_res: bool = True,
    cv_tol: float = 1e-3,
    single_instance: bool = True,
):
    model.eval()
    opt_obj, hcnn_obj, eq_cv, ineq_cv = [], [], [], []

    with torch.no_grad():
        for X, obj in loader:
            X, obj = X.to(device), obj.to(device)
            X_full = torch.cat(
                (X, torch.zeros(X.shape[0], A.shape[1] - X.shape[1], 1, device=device)), dim=1
            )
            predictions = model(X[:, :, 0], X_full, test=True)

            opt_obj.append(obj)
            hcnn_obj.append(batched_objective(predictions))

            # Equality constraint violation
            eq_cv_batch = torch.abs(
                torch.matmul(A[0].reshape(1, A.shape[1], A.shape[2]), predictions.reshape(X.shape[0], A.shape[2], 1))
                - X_full
            ).amax(dim=1)
            eq_cv.append(eq_cv_batch)

            # Inequality violation
            ineq_cv_batch_ub = torch.clamp(predictions.reshape(X.shape[0], A.shape[2], 1) - ub, min=0)
            ineq_cv_batch_lb = torch.clamp(lb - predictions.reshape(X.shape[0], A.shape[2], 1), min=0)
            ineq_cv_batch = torch.maximum(ineq_cv_batch_ub, ineq_cv_batch_lb) / ub
            ineq_cv.append(ineq_cv_batch.amax(dim=1))

    # Aggregate
    opt_obj = torch.cat(opt_obj, dim=0)
    hcnn_obj = torch.cat(hcnn_obj, dim=0)
    eq_cv = torch.cat(eq_cv, dim=0)
    ineq_cv = torch.cat(ineq_cv, dim=0)

    opt_obj_mean = opt_obj.mean().item()
    hcnn_obj_mean = hcnn_obj.mean().item()
    eq_cv_mean, eq_cv_max = eq_cv.mean().item(), eq_cv.max().item()
    ineq_cv_mean, ineq_cv_max = ineq_cv.mean().item(), ineq_cv.max().item()
    ineq_perc = (1 - (ineq_cv > cv_tol).float().mean().item()) * 100.0

    # Inference time
    model_input = X[:1, :, :] if single_instance else X
    X_full_inf = torch.cat(
        (model_input, torch.zeros(model_input.shape[0], A.shape[1] - model_input.shape[1], 1, device=device)), dim=1
    )

    times = timeit.repeat(
        lambda: model(model_input[:, :, 0], X_full_inf, test=True),
        repeat=time_evals,
        number=1,
    )
    eval_time, eval_time_std = np.mean(times), np.std(times)

    if print_res:
        print(f"=========== {prefix} performance ===========")
        print(f"Mean objective                : {hcnn_obj_mean:.5f}")
        print(f"Mean|Max eq. cv               : {eq_cv_mean:.5f} | {eq_cv_max:.5f}")
        print(f"Mean|Max normalized ineq. cv  : {ineq_cv_mean:.5f} | {ineq_cv_max:.5f}")
        print(f"Perc of valid cv. tol.        : {ineq_perc:.3f}%")
        print(f"Time for evaluation [s]       : {eval_time:.5f}")
        print(f"Optimal mean objective        : {opt_obj_mean:.5f}")

    return opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, eval_time, eval_time_std


# ============================================================
#   MAIN (PyTorch)
# ============================================================

def main(
    filepath: str,
    config_path: str,
    SEED: int,
    PLOT_TRAINING: bool,
    SAVE_RESULTS: bool,
    use_saved: bool,
    results_folder: str | None,
    run_name: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    torch.manual_seed(SEED)

    hyperparameters = load_configuration(config_path)
    
    (
        As,
        lbxs,
        ubxs,
        lbus,
        ubus,
        xhat,
        alpha,
        T,
        base_dim,
        X,
        train_loader,
        valid_loader,
        test_loader,
        batched_objective,
    ) = load_data(filepath=filepath, batch_size=hyperparameters["batch_size"])

    # Move tensors to device
    As, lbxs, ubxs, lbus, ubus, X = [t.float().to(device) for t in [As, lbxs, ubxs, lbus, ubus, X]]

    Y_DIM = As.shape[2]
    X_full = torch.cat((X, torch.zeros(X.shape[0], As.shape[1] - X.shape[1], 1, device=device)), dim=1)
    lb = torch.cat((lbxs, lbus), dim=1)
    ub = torch.cat((ubxs, ubus), dim=1)

    # Initialize model
    model, optimizer, train_step = setup_model(
        hyperparameters=hyperparameters,
        A=As,
        X=X,
        b=X_full,
        lb=lb,
        ub=ub,
        batched_objective=batched_objective,
        device=device,
    )

    if use_saved:
        if results_folder is None:
            raise ValueError("Please provide the results folder name to load saved parameters.")
        # Load saved model
        model_path = pathlib.Path(__file__).parent / "results" / results_folder / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        n_epochs = hyperparameters["n_epochs"]
        train_losses, val_losses, eqcvs, ineqcvs = [], [], [], []

        with (
            Logger(run_name=run_name, project_name="hcnn_toy_mpc") as data_logger,
            GracefulShutdown("Stop detected, finishing epoch...") as g,
        ):
            data_logger.run.config.update(hyperparameters)

            for epoch in (pbar := tqdm(range(n_epochs))):
                if g.stop:
                    break
                model.train()
                epoch_losses, batch_sizes = [], []

                start_epoch_time = time.time()
                for X_batch, _ in train_loader:
                    X_batch = X_batch.to(device)
                    X_batch_full = torch.cat(
                        (X_batch, torch.zeros(X_batch.shape[0], As.shape[1] - X_batch.shape[1], 1, device=device)), dim=1
                    )
                    loss = train_step(model, optimizer, X_batch[:, :, 0], X_batch_full)
                    epoch_losses.append(loss)
                    batch_sizes.append(X_batch.shape[0])

                weighted_epoch_loss = sum(l * s for l, s in zip(epoch_losses, batch_sizes)) / sum(batch_sizes)
                train_losses.append(weighted_epoch_loss)
                epoch_time = time.time() - start_epoch_time
                pbar.set_description(f"Train Loss: {weighted_epoch_loss:.5f}")

                # Validation per epoch
                model.eval()
                with torch.no_grad():
                    for X_valid, valid_obj in valid_loader:
                        X_valid = X_valid.to(device)
                        valid_obj = valid_obj.to(device)
                        X_valid_full = torch.cat(
                            (X_valid, torch.zeros(X_valid.shape[0], As.shape[1] - X_valid.shape[1], 1, device=device)),
                            dim=1
                        )
                        predictions = model(X_valid[:, :, 0], X_valid_full, test=True)
                        val_loss = batched_objective(predictions)
                        eqcv = torch.abs(As[0] @ predictions.reshape(-1, Y_DIM, 1) - X_valid_full).amax()
                        pred_reshaped = predictions.reshape(-1, Y_DIM, 1)
                        ineqcv = torch.maximum(torch.amax(pred_reshaped - ub, dim=1),
                                               torch.amax(lb - pred_reshaped, dim=1)).mean()

                        data_logger.log(epoch, {
                            "weighted_epoch_loss": weighted_epoch_loss,
                            "epoch_training_time": epoch_time,
                            "validation_objective_mean": val_loss.mean().item(),
                            "validation_average_rs": ((val_loss - valid_obj) / valid_obj.abs()).mean().item(),
                            "validation_cv": torch.max(eqcv, ineqcv).item(),
                            "validation_time": time.time() - start_epoch_time,
                        })

                        pbar.set_postfix({
                            "eqcv": f"{eqcv.item():.5f}",
                            "ineqcv": f"{ineqcv.item():.5f}",
                            "Valid. Loss": f"{val_loss.mean().item():.5f}"
                        })

        if SAVE_RESULTS:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = pathlib.Path(__file__).parent / "results" / timestamp
            results_folder.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), results_folder / "model.pt")
            np.savez(results_folder / "results.npz", train_losses=train_losses, val_losses=val_losses)

    # Final evaluation
    opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, mean_inf_time, std_inf_time = evaluate_hcnn(
        test_loader, model, batched_objective, As, lb, ub, "Test", device
    )

    _, _, _, _, _, mean_inf_time_single, std_inf_time_single = evaluate_hcnn(
        test_loader, model, batched_objective, As, lb, ub, "Test", device, single_instance=True
    )

    # Log summary metrics
    rs = (hcnn_obj - opt_obj) / torch.abs(opt_obj)
    cv = torch.max(eq_cv, ineq_cv)
    cv_thres = 1e-3
    with Logger(run_name=run_name, project_name="hcnn_toy_mpc") as data_logger:
        data_logger.run.summary.update({
            "Average RS Test": rs.mean().item(),
            "Max CV Test": cv.max().item(),
            "Percentage CV < tol": (1 - (cv > cv_thres).float().mean().item()) * 100,
            "Average Single Inference Time": mean_inf_time_single,
            "Average Batch Inference Time": mean_inf_time,
        })

    if PLOT_TRAINING and not use_saved:
        plot_training(train_loader, valid_loader, train_losses, val_losses, eqcvs, ineqcvs)

    return model


# ============================================================
#   SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HCNN on toy MPC problem (PyTorch version, GPU ready).")
    parser.add_argument("--filename", type=str, required=True, help="Filename of dataset (.npz)")
    parser.add_argument("--config", type=str, default="toy_MPC", help="Configuration file for HCNN hyperparameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot_training", action=argparse.BooleanOptionalAction, default=False, help="Plot training curves")
    parser.add_argument("--save_results", action=argparse.BooleanOptionalAction, default=True, help="Save results")
    parser.add_argument("--use_saved", action="store_true", help="Use saved network to evaluate")
    parser.add_argument("--results_folder", type=str, default=None, help="Folder containing saved model/results")
    args = parser.parse_args()

    run_name = f"toy_MPC_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    config_path = pathlib.Path(__file__).parent.parent.resolve() / "configs" / (args.config + ".yaml")

    main(
        filepath=args.filename,
        config_path=config_path,
        SEED=args.seed,
        PLOT_TRAINING=args.plot_training,
        SAVE_RESULTS=args.save_results,
        use_saved=args.use_saved,
        results_folder=args.results_folder,
        run_name=run_name,
    )