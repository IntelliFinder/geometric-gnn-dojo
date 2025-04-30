import torch
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
import argparse
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import sys
# Import EGNN model from the original code
from models.egnn import EGNNModel

def orthogonalize(U):
    """
    Orthogonalize a set of linear independent vectors using Gram–Schmidt process.
    Matches the implementation from the original code.
    """
    Q, R = torch.linalg.qr(U)
    S = torch.sign(torch.diag(R))
    return Q * S


def compute_eigenspace_projectors(E, U):
    """
    Compute eigenspace projectors for each eigenvalue.
    
    Args:
        E: Eigenvalues tensor
        U: Eigenvectors tensor
        
    Returns:
        eigenspaces: Dictionary mapping eigenvalues to their projector matrices
        sorted_eigenvals: Sorted list of eigenvalues (largest to smallest)
    """
    eigenspaces = {}
    unique_eigenvalues = torch.unique(E.round(decimals=10))
    
    for eigenval in unique_eigenvalues:
        # Find indices of eigenvectors corresponding to this eigenvalue
        indices = torch.where(torch.isclose(E, eigenval, rtol=1e-8))[0]
        
        # Get the eigenvectors for this eigenspace
        eigenvectors = U[:, indices]
        
        # Compute projector matrix for this eigenspace: P = U_λ U_λ^T
        projector = eigenvectors @ eigenvectors.T
        
        # Store in dictionary with eigenvalue as key
        eigenspaces[eigenval.item()] = projector
    
    # Sort eigenvalues in descending order (largest first)
    sorted_eigenvals = sorted(eigenspaces.keys(), reverse=True)
    
    return eigenspaces, sorted_eigenvals



def create_edge_features_with_k_projectors(edges, E, U, k=10):
    """
    Create edge features by stacking the k eigenspace projectors.
    
    Args:
        edges: Edge indices [2, num_edges]
        E: Eigenvalues tensor
        U: Eigenvectors tensor
        k: Number of top eigenvalues to consider
        
    Returns:
        edge_attr: Edge features tensor [num_edges, k]
    """
    device = edges.device
    num_edges = edges.shape[1]
    
    # Compute eigenspace projectors and get sorted eigenvalues
    eigenspaces, sorted_eigenvals = compute_eigenspace_projectors(E, U)
    
    # Limit to k largest eigenvalues, or all if fewer than k
    k = min(k, len(sorted_eigenvals))
    top_k_eigenvals = sorted_eigenvals[:k]
    
    # Get source and target nodes for each edge
    row, col = edges
    
    # Initialize edge features - we'll have k features per edge
    edge_attr = torch.zeros((num_edges, k), dtype=torch.float64, device=device)
    
    # Stack projector matrices for top-k eigenvalues
    stacked_projectors = []
    for eigenval in top_k_eigenvals:
        stacked_projectors.append(eigenspaces[eigenval])
    
    # Create a tensor of shape [k, num_nodes, num_nodes]
    stacked_projectors = torch.stack(stacked_projectors).permute(1, 2, 0)
    edge_attr = stacked_projectors[row, col]
    
    return edge_attr

def epnn_forward(batch, egnn_base, k_projectors=10):
    """
    Eliminating sign ambiguity of the input eigenvectors with EPNN.
    Uses k largest eigenvalues and their projectors as edge features.
    """
    n, d = batch.pos.shape
    device = batch.pos.device
    
    x_upd = egnn_base(batch)
    assert not torch.allclose(x_upd, batch.pos.to(x_upd.device)), "EGNN output equals the eigenvectors input" 

    # Calculate signs based on sum
    sums = torch.sum(x_upd, dim=0).round(decimals=6)
    if sums.dim() > 1:
        sums = sums.squeeze()
    
    sums = torch.sign(sums).to(device)
    sums[sums == 0] = 1  # Set the sign of zero to 1
    
    # Create diagonal matrix of signs
    sums = torch.diag(sums)
    
    # Apply the sign correction (matrix multiplication as in original)
    U_canonical = batch.pos @ sums.t()
    
    return U_canonical, x_upd


def count_canonicalization(pyg_data, egnn_base, k_projectors=10):
    """Count canonicalizable eigenvectors using different methods"""
    # Create edge structure
    edges = pyg_data.edge_index
    # Get adjacency matrix in similar format to original code
    n = pyg_data.num_nodes
    
    # Create adjacency matrix from edge_index
    row, col = edges
    A = torch.zeros((n, n), device=edges.device)
    A[row, col] = 1
    
    # Compute normalized adjacency (similar to original code)
    D_inv_sqrt = torch.diag(torch.sum(A, dim=1).clip(1) ** -0.5)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute eigendecomposition of A_norm (for consistency with original code)
    E, U = torch.linalg.eigh(A_norm)
    
    # Round eigenvalues to avoid numerical instability
    E = E.round(decimals=14)
    
    # Exclude the trivial eigenvector (last one for normalized adjacency)
    
    E, U = torch.flip(E[:-1], dims=[0]), torch.flip(U[:, :-1], dims=[-1])
    E, U = E[:k_projectors], U[:, :k_projectors]
    pyg_data.pos = U  # Use eigenvectors as node positions
    pyg_data.eigvals = E  # Store eigenvalues
    pyg_data.x = torch.zeros((n, args.in_dim), dtype=torch.long,device=edges.device)  # Dummy node features
    #pyg_data.edge_attr = create_edge_features_with_k_projectors(edges, E, U, k=k_projectors)
    if U.size(1) <  k_projectors:
        wrapper = torch.zeros((pyg_data.pos.size(0), k_projectors), dtype=torch.float64, device=edges.device)
        wrapper[:, :pyg_data.pos.size(1)] = pyg_data.pos
        pyg_data.pos = wrapper
    # Set the eigenvector dimension (number of columns in U)
    
    #dataset = [pyg_data]  # Wrap in a list for DataLoader
    #batch = Batch.from_data_list(dataset)
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)
    batch = pyg_data
    # Count unique eigenvalues and their multiplicities
    _, mult = torch.unique(E, return_counts=True)
    
    # Find indices of eigenvectors with multiplicity 1
    single_ind = torch.where(mult == 1)[0]
    
    # Compute sums of each eigenvector (simple method)
    orig_sums = torch.sum(U, dim=0).abs().round(decimals=6)
    non_zeros_orig = torch.count_nonzero(orig_sums[single_ind])
    num_uncan_orig = single_ind.size(0) - non_zeros_orig
    
    # If we're just initializing, return early
    if egnn_base is None:
        return num_uncan_orig, 0, len(single_ind), n
    # Process with enhanced EGNN (using k projectors)
    U_canonical, x_upd = epnn_forward(batch, egnn_base, k_projectors=k_projectors)
    
    # Check EGNN canonicalization
    egnn_sums = torch.sum(x_upd, dim=0).abs().round(decimals=6)
    non_zeros = torch.count_nonzero(egnn_sums[single_ind])
    num_uncan = single_ind.size(0) - non_zeros
    
    return num_uncan_orig.item(), num_uncan.item(), len(single_ind), n


def main():
    # Set defaults using similar arguments as the original code
    parser = argparse.ArgumentParser(description='ZINC Canonicalization Test with PyTorch Geometric')
    parser.add_argument('--num_layers', type=int, default=5, help='number of message passing layers')
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--in_dim', type=int, default=128, help='input feature dimension')
    parser.add_argument('--coords_weight', type=float, default=3.0, help='coordinate update weight')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'silu', 'leakyrelu'], help='activation function')
    parser.add_argument('--norm', type=str, default='layer', choices=['layer', 'batch', 'none'], help='normalization type')
    parser.add_argument('--aggr', type=str, default='sum', choices=['sum', 'mean', 'max'], help='aggregation function')
    parser.add_argument('--residual', type=bool, default=False, help='use residual connections')
    parser.add_argument('--subset_size', type=int, default=100, help='number of ZINC graphs to test')
    parser.add_argument('--k_projectors', type=int, default=10, help='number of top eigenvalue projectors to use')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    global args
    args = parser.parse_args()

    
    # Set default precision
    torch.set_default_dtype(torch.float64)
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Calculate the edge feature dimension: k eigenvalues + k projector entries
    edge_attr_dim = 2 * args.k_projectors
    print(f"Using {args.k_projectors} top eigenvalues and their projectors as edge features.")
    

    # Now we can initialize the model with the correct output dimension
    # Initialize EGNN model with the provided EGNNModel class
    egnn_base = EGNNModel(
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        proj_dim=args.k_projectors,
    ).to(device)


    # Load ZINC dataset from PyTorch Geometric
    print("Loading ZINC dataset from PyTorch Geometric...")
    transform = None
    dataset = ZINC(root='./data/ZINC', subset=True, transform=transform)

    # Limit to subset for testing
    subset_size = min(args.subset_size, len(dataset))
    dataset = dataset[:subset_size]
    
    # Process graphs
    mols_sign_uncano, epnn_sign_uncano, total_eigvecs, total_nodes = 0, 0, 0, 0
    
    print(f"Processing {len(dataset)} graphs from ZINC subset...")
    for data in tqdm(dataset):
        data = data.to(device)
        #try:
        # Check if the graph is connected
        if data.num_nodes == 0 or data.edge_index.size(1) == 0:
            continue
        # Count canonicalization
        sign_uncan, epnn_uncan, n_eigvecs, n_nodes = count_canonicalization(
            data, egnn_base, k_projectors=args.k_projectors
        )
        #except Exception as e:
        #    print(f"Error processing graph: {e}")
        #    continue

        mols_sign_uncano += sign_uncan
        epnn_sign_uncano += epnn_uncan
        total_eigvecs += n_eigvecs
        total_nodes += n_nodes
    
    # Print results in the same format as the original code
    print("\nResults:")
    print("'%' of sign uncanonicalizable is: " + str(100*mols_sign_uncano/total_eigvecs))
    print("'%' of EPNN uncanonicalizable is: " + str(100*epnn_sign_uncano/total_eigvecs))
    print("Improvement of EGNN over standard sign: " + str(100*(mols_sign_uncano-epnn_sign_uncano)/total_eigvecs) + "%")
    
    # Create a direct implementation of the sign canonicalization
    def direct_sign_canon(U):
        col_sums = torch.sum(U, dim=0)
        col_signs = torch.sign(col_sums)
        col_signs[col_signs == 0] = 1  # Handle zero sums
        sign_matrix = torch.diag(col_signs)
        U_canonical = U @ sign_matrix
        return U_canonical
    

    print("\nRecommendation:")
    if epnn_sign_uncano < mols_sign_uncano:
        print(f"The EGNN approach with {args.k_projectors} direct eigenvalues and projectors improved sign canonicalization.")
        print("You can use this approach with adjusted parameters.")
    else:
        print(f"The EGNN with {args.k_projectors} direct eigenvalues and projectors did not significantly improve sign canonicalization.")
        print("Use the direct sign canonicalization approach:")
        print("""
def sign_canonicalization(U):
    # Calculate column-wise sums to determine sign direction
    col_sums = torch.sum(U, dim=0)
    col_signs = torch.sign(col_sums)
    col_signs[col_signs == 0] = 1  # Handle zero sums
    
    # Apply sign correction to get canonical form
    sign_matrix = torch.diag(col_signs)
    U_canonical = U @ sign_matrix
    
    return U_canonical
        """)


if __name__ == "__main__":
    main()
