from enum import Enum
import pandas as pd
from typing import List, Tuple, Optional
from flyvis.network import Network


class PerturbationType(Enum):
    PAIR_WISE = 'pair-wise'
    MOTIF = 'motif'


class FlyvisCellTypePert:
    def perturb(self, conn: pd.DataFrame, perturbation_type: PerturbationType,
                pairs: Optional[List[Tuple[str, str]]] = None,
                motif_id: Optional[str] = None) -> pd.DataFrame:
        """Perturb a connectivity dataframe based on the specified perturbation type.

        Args:
            conn: DataFrame with 'source_type' and 'target_type' columns representing cell type connections
            perturbation_type: Type of perturbation to apply
            pairs: List of tuples (pre, post) representing cell type pairs to perturb (for PAIR_WISE)
            motif_id: Motif identifier string (for MOTIF)
        Returns:
            DataFrame with added 'pert_weight' column
        """
        result = conn.copy()

        if perturbation_type == PerturbationType.PAIR_WISE:
            if pairs is None:
                raise ValueError("pairs parameter is required for PAIR_WISE perturbation")
            return self._perturb_pair_wise(result, pairs)
        elif perturbation_type == PerturbationType.MOTIF:
            if motif_id is None:
                raise ValueError("motif_id parameter is required for MOTIF perturbation")
            return self._perturb_motif(result, motif_id)

        return result

    def _perturb_pair_wise(self, conn: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Apply pair-wise perturbation by setting weights to 0 for specified pairs,
        and weight 1 for all other pairs."""

        conn['pert_weight'] = 1
        for source_type, target_type in pairs:
            mask = (conn['source_type'] == source_type) & (conn['target_type'] == target_type)
            conn.loc[mask, 'pert_weight'] = 0
        return conn

    def _perturb_motif(self, conn: pd.DataFrame, motif_id: str) -> pd.DataFrame:
        """Apply motif-based perturbation using the specified motif identifier."""
        # TODO: Implement motif perturbation
        pass

    def override_network(self, network: Network, pert_conn: pd.DataFrame):
        """Override network synaptic strengths by multiplying with perturbation weights.
        Args:
            network: Network object with edge_params.syn_strength containing keys and raw_values
            pert_conn: DataFrame with 'source_type', 'target_type', and 'pert_weight' columns
        """
        syn_str = network.edge_params.syn_strength.keys

        for idx, (src, tar) in enumerate(syn_str):
            # Find matching row in pert_conn for this (source_type, target_type) pair
            mask = (pert_conn['source_type'] == src) & (pert_conn['target_type'] == tar)
            matching_rows = pert_conn[mask]

            if not matching_rows.empty:
                pert_weight = matching_rows['pert_weight'].iloc[0]
                network.edge_params.syn_strength.raw_values.detach()[idx] *= pert_weight



if __name__ == '__main__':
    conn_data = {
        'source_type': ['C1', 'L1', 'Am', 'T1', 'C1', 'L1'],
        'target_type': ['L1', 'Am', 'T1', 'T2', 'T2', 'T1']
    }
    conn_df = pd.DataFrame(conn_data)

    print("Original connectivity dataframe:")
    print(conn_df)
    print()

    # Create perturbation instance
    pert = FlyvisCellTypePert()

    # Perturb two pairs: (C1, L1) and (Am, T1)
    pairs_to_perturb = [('C1', 'L1'), ('Am', 'T1')]
    result_df = pert.perturb(conn_df, PerturbationType.PAIR_WISE, pairs=pairs_to_perturb)

    print(f"Perturbed dataframe (pairs {pairs_to_perturb} set to weight 0):")
    print(result_df)
