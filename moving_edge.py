from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.animations.hexscatter import HexScatter
import numpy as np

import os
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from flyvis import NetworkView
from flyvis.datasets.rendering import BoxEye
from flyvis.analysis.animations import HexScatter
import h5py
import datamate.io        # Where the function is defined
import datamate.directory # Where the function is actually CALLED causing the error
import pandas as pd

from flyvis_cell_type_pert import FlyvisCellTypePert, PerturbationType
from flyvis.analysis.moving_bar_responses import preferred_direction
from flyvis.analysis.moving_bar_responses import dsi_correlation_to_known
from flyvis.analysis.moving_bar_responses import direction_selectivity_index
from flyvis.analysis.moving_bar_responses import correlation_to_known_tuning_curves

data_path = Path("data/flyvis_data")
data_path.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["FLYVIS_ROOT_DIR"] = str(data_path)

def fixed_write_h5(path, val):
    """
    A Windows-safe replacement that skips the 'read-before-write' check.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Simple, safe write. No checking 'f["data"]', no 'unlink'.
    # This completely bypasses the logic causing your KeyError.
    with h5py.File(path, mode="w", libver="latest") as f:
        f.create_dataset("data", data=val)
datamate.io._write_h5 = fixed_write_h5
if hasattr(datamate.directory, "_write_h5"):
    datamate.directory._write_h5 = fixed_write_h5
    print(" -> Patched datamate.directory._write_h5")
else:
    print(" -> Warning: Could not find _write_h5 in directory module")

print("Importing flyvis...")
from flyvis import NetworkView
import shutil


class MovingEdgeWrapper():
    def __init__(self,
                dataset: MovingEdge,
                 pert_folder_name=None,
                 pert: FlyvisCellTypePert = None,
                 output_file_name = ''):
        self.dataset = dataset
        self.src_folder = data_path / "results/flow/0000/000"

        if pert_folder_name is not None:
            self.tar_folder = data_path / f"results/flow/0000/000_{pert_folder_name}"
            # Clean start: remove target folder if it exists
            shutil.rmtree(self.tar_folder, ignore_errors=True)
            # Copy original to target
            shutil.copytree(self.src_folder, self.tar_folder, dirs_exist_ok=True)
        else:
            self.tar_folder = self.src_folder

        self.output_file_name = output_file_name
        self.network_view = NetworkView(self.tar_folder)
        self.network = self.network_view.init_network()
        # self.cell_type_df = pd.read_csv(f'{data_path}/flyvis_cell_type_connectivity.csv')
        self.pert = pert


    def run(self):
        print('Running moving edge simulation...')

        if self.pert is not None:
            print('Applying perturbation to network in memory...')
            # 1. Perturb the in-memory network
            self.pert.override_network(self.network)

            # 2. SAVE PERTURBED WEIGHTS TO DISK (The Friend's Fix)
            print("Overwriting disk checkpoints with perturbed weights...")

            # Load the original checkpoint template structure
            # We use map_location='cpu' to be safe
            checkpoint_template = torch.load(self.src_folder / "best_chkpt", map_location='cpu')

            # Create a copy and update the network weights with our perturbed ones
            perturbed_checkpoint = checkpoint_template.copy()
            perturbed_checkpoint['network'] = self.network.state_dict()

            # Save to 'best_chkpt' in the TARGET folder
            target_best_chkpt = self.tar_folder / "best_chkpt"
            torch.save(perturbed_checkpoint, target_best_chkpt)
            print(f" -> Updated: {target_best_chkpt}")

            # Also overwrite ALL checkpoints in the 'chkpts' folder
            # Flyvis sometimes loads specific epochs, so we ensure they are all perturbed
            chkpts_dir = self.tar_folder / "chkpts"
            if chkpts_dir.exists():
                for chkpt_file in chkpts_dir.glob("*"):
                    torch.save(perturbed_checkpoint, chkpt_file)
                    print(f" -> Updated: {chkpt_file}")

            # 3. CLEAR CACHE (Crucial for Flyvis to see the change)
            print("Clearing caches...")
            for cache_name in ["__cache__", "__storage__"]:
                cache_dir = self.tar_folder / cache_name
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    print(f" -> Removed {cache_name}")

            # Re-init network view to ensure it sees the changes on disk
            self.network_view = NetworkView(self.tar_folder)

        print('Generating moving edge responses...')
        stims_and_resps = self.network_view.moving_edge_responses(self.dataset)

        print('Evaluating performance...')

        pds = preferred_direction(stims_and_resps)
        dsis = direction_selectivity_index(stims_and_resps)
        cell_types_to_collect = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
        data = []

        for cell_type in cell_types_to_collect:
            pd1 = pds.custom.where(cell_type=cell_type, intensity=1).item()
            pd0 = pds.custom.where(cell_type=cell_type, intensity=0).item()

            dsi1 = dsis.custom.where(cell_type=cell_type, intensity=1).item()
            dsi0 = dsis.custom.where(cell_type=cell_type, intensity=0).item()

            dsi_corr = dsi_correlation_to_known(direction_selectivity_index(stims_and_resps)).median() # TODO; ask Zina
            corrs = correlation_to_known_tuning_curves(stims_and_resps)
            tc_corr1 = corrs.custom.where(cell_type=cell_type, intensity=1)
            tc_corr0 = corrs.custom.where(cell_type=cell_type, intensity=0)

            data.append({
                'cell_type': cell_type,
                'pd_intensity1': pd1,
                'pd_intensity0': pd0,
                'dsi_intensity1': dsi1,
                'dsi_intensity0': dsi0,
                'dsi_correlation': dsi_corr.item(),
                'tc_correlation_intensity1': tc_corr1.item(),
                'tc_correlation_intensity0': tc_corr0.item()
            })

        results_df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        results_df.to_csv(self.output_file_name, index=False)



if __name__ == "__main__":
    dataset = MovingEdge(
            offsets=[-10, 11],  # offset of bar from center in 1 * radians(2.25) led size
            intensities=[0, 1],  # intensity of bar
            speeds=[19],  # speed of bar in 1 * radians(5.8) / s
            height=80,  # height of moving bar in 1 * radians(2.25) led size
            post_pad_mode="continue",  # for post-stimulus period, continue with the last frame of the stimulus
            t_pre=1.0,  # duration of pre-stimulus period
            t_post=1.0,  # duration of post-stimulus period
            dt=1 / 200,  # temporal resolution of rendered video
            angles=list(np.arange(0, 360, 30)),  # motion direction (orthogonal to edge)
    )


    wrapper = MovingEdgeWrapper(dataset, pert=None, pert_folder_name=None,
                                output_file_name="data/flyvis_data/perf/original_network.csv")
    wrapper.run()


    conn_df = pd.read_csv('data/flyvis_data/flyvis_cell_type_connectivity.csv')
    pert = FlyvisCellTypePert()
    pairs_to_perturb = [('L4', 'L4')]

    # motif_id = "78 ['+', '+', '-', '-']"
    # pert.perturb(conn_df, PerturbationType.MOTIF, motif_id=str(motif_id))
    pert.perturb(conn_df, PerturbationType.PAIR_WISE, pairs=pairs_to_perturb)
    print(pert.pert_conn[pert.pert_conn.pert_weight == 0])

    wrapper = MovingEdgeWrapper(dataset, pert=pert, pert_folder_name='test3',
                                output_file_name="data/flyvis_data/perf/pairwise-L4-L4-pert.csv")
    wrapper.run()
