from flyvis_cell_type_pert import FlyvisCellTypePert, PerturbationType
from flyvis.datasets.sintel import MultiTaskSintel
from pathlib import Path
import os
import h5py
import datamate
import pandas as pd
import numpy as np
import torch
import shutil

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


class SintelWrapper():
    def __init__(self,
                 dataset: MultiTaskSintel,
                 pert_folder_name=None,
                 pert: FlyvisCellTypePert = None,
                 output_file_name=''):
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
        self.pert = pert

    def run(self):
        print('Running Sintel optic flow simulation...')

        if self.pert is not None:
            print('Applying perturbation to network in memory...')
            # 1. Perturb the in-memory network
            self.pert.override_network(self.network)

            # 2. SAVE PERTURBED WEIGHTS TO DISK
            print("Overwriting disk checkpoints with perturbed weights...")

            checkpoint_template = torch.load(self.src_folder / "best_chkpt", map_location='cpu')
            perturbed_checkpoint = checkpoint_template.copy()
            perturbed_checkpoint['network'] = self.network.state_dict()

            target_best_chkpt = self.tar_folder / "best_chkpt"
            torch.save(perturbed_checkpoint, target_best_chkpt)
            print(f" -> Updated: {target_best_chkpt}")

            chkpts_dir = self.tar_folder / "chkpts"
            if chkpts_dir.exists():
                for chkpt_file in chkpts_dir.glob("*"):
                    torch.save(perturbed_checkpoint, chkpt_file)
                    print(f" -> Updated: {chkpt_file}")

            # 3. CLEAR CACHE
            print("Clearing caches...")
            for cache_name in ["__cache__", "__storage__"]:
                cache_dir = self.tar_folder / cache_name
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    print(f" -> Removed {cache_name}")

            self.network_view = NetworkView(self.tar_folder)
            self.network = self.network_view.init_network()

        print('Initializing decoder...')
        # Initialize the decoder
        decoder = self.network_view.init_decoder()["flow"]
        decoder.eval()  # Set to evaluation mode

        print('Generating Sintel optic flow responses...')
        
        # Collect predictions and ground truth for all sequences
        all_pred_flow = []
        all_true_flow = []
        all_epe = []
        
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            lum = data["lum"]
            flow = data["flow"]
            
            # Simulate network response
            stationary_state = self.network.fade_in_state(1.0, self.dataset.dt, lum[[0]])
            responses = self.network.simulate(lum[None], self.dataset.dt, initial_state=stationary_state)
            
            # Decode flow from neural responses
            y_pred = decoder(responses)
            
            # Compute EPE for this sequence
            epe = torch.sqrt(((y_pred - flow) ** 2).sum(dim=1))  # EPE per frame
            
            all_pred_flow.append(y_pred.detach().cpu())
            all_true_flow.append(flow.cpu() if hasattr(flow, 'cpu') else flow)
            all_epe.append(epe.detach().cpu())
        
        print('Evaluating performance...')
        
        # Aggregate metrics
        all_epe_tensor = torch.cat(all_epe, dim=0)  # Concatenate all EPE values
        
        # Compute overall statistics
        data = []
        
        data.append({
            'sequence': 'overall',
            'n_sequences': len(self.dataset),
            'mean_epe': float(all_epe_tensor.mean()),
            'median_epe': float(all_epe_tensor.median()),
            'std_epe': float(all_epe_tensor.std()),
            'epe_pixel_1': float((all_epe_tensor < 1).float().mean()),  # % pixels with EPE < 1
            'epe_pixel_3': float((all_epe_tensor < 3).float().mean()),  # % pixels with EPE < 3
            'epe_pixel_5': float((all_epe_tensor < 5).float().mean()),  # % pixels with EPE < 5
        })
        
        # Per-sequence statistics
        for i, epe in enumerate(all_epe):
            data.append({
                'sequence': f'seq_{i:03d}',
                'sequence_name': self.dataset.arg_df.iloc[i]['name'] if hasattr(self.dataset, 'arg_df') else f'seq_{i}',
                'mean_epe': float(epe.mean()),
                'median_epe': float(epe.median()),
                'std_epe': float(epe.std()),
            })

        results_df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        results_df.to_csv(self.output_file_name, index=False)
        
        print(f"\nResults saved to {self.output_file_name}")
        overall = results_df[results_df['sequence'] == 'overall'].iloc[0]
        print(f"Mean EPE: {overall['mean_epe']:.4f} pixels")
        print(f"Median EPE: {overall['median_epe']:.4f} pixels")
        print(f"% pixels with EPE < 3px: {overall['epe_pixel_3']*100:.2f}%")


if __name__ == "__main__":
    # Initialize Sintel dataset using the exact parameters from the documentation
    print("Initializing Sintel dataset...")
    
    dataset = MultiTaskSintel(
        tasks=["flow"],
        boxfilter=dict(extent=15, kernel_size=13),
        vertical_splits=1,  # Can use 3 for more augmentation, but 1 for faster testing
        n_frames=19,
        dt=1/50,  # Temporal resolution
        augment=False,  # Set to False for evaluation
        resampling=True,
        interpolate=True,
        all_frames=False,
        random_temporal_crop=False,
    )
    
    print(f"Dataset initialized with {len(dataset)} sequences")
    if hasattr(dataset, 'arg_df'):
        print(f"Sequences: {dataset.arg_df['name'].tolist()[:5]}... (showing first 5)")

    # Run original network
    print("\n" + "="*60)
    print("RUNNING ORIGINAL NETWORK")
    print("="*60)
    wrapper = SintelWrapper(
        dataset, 
        pert=None, 
        pert_folder_name=None,
        output_file_name="data/flyvis_data/perf/sintel_original_network.csv"
    )
    wrapper.run()

    # Run perturbed network
    print("\n" + "="*60)
    print("RUNNING PERTURBED NETWORK (L4-L4)")
    print("="*60)
    conn_df = pd.read_csv('data/flyvis_data/flyvis_cell_type_connectivity.csv')
    pert = FlyvisCellTypePert()
    pairs_to_perturb = [('L4', 'L4')]

    pert.perturb(conn_df, PerturbationType.PAIR_WISE, pairs=pairs_to_perturb)
    print("\nPerturbed connections:")
    print(pert.pert_conn[pert.pert_conn.pert_weight == 0])

    wrapper = SintelWrapper(
        dataset, 
        pert=pert, 
        pert_folder_name='sintel_L4_L4_pert',
        output_file_name="data/flyvis_data/perf/sintel_pairwise-L4-L4-pert.csv"
    )
    wrapper.run()
    
    print("\n" + "="*60)
    print("DONE - Compare results in data/flyvis_data/perf/")
    print("="*60)