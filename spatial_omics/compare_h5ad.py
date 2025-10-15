"""
Compare two h5ad files and report their differences.

This script compares the structure, dimensions, and content of two AnnData objects
stored in h5ad format.

Usage:
    python compare_h5ad.py <file1.h5ad> <file2.h5ad> [--detailed]
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import anndata as ad
except ImportError:
    print("Error: anndata package is required. Install it with: pip install anndata")
    sys.exit(1)


def format_shape(shape):
    """Format shape tuple as string."""
    return f"{shape[0]} × {shape[1]}" if len(shape) == 2 else str(shape)


def compare_basic_info(adata1, adata2):
    """Compare basic information of two AnnData objects."""
    print("\n" + "="*80)
    print("BASIC INFORMATION COMPARISON")
    print("="*80)
    
    info = {
        "Shape": (adata1.shape, adata2.shape),
        "n_obs (cells/spots)": (adata1.n_obs, adata2.n_obs),
        "n_vars (genes/features)": (adata1.n_vars, adata2.n_vars),
    }
    
    for key, (val1, val2) in info.items():
        match = "✓" if val1 == val2 else "✗"
        if isinstance(val1, tuple):
            print(f"{match} {key:25s}: {format_shape(val1):15s} vs {format_shape(val2):15s}")
        else:
            print(f"{match} {key:25s}: {val1:15} vs {val2:15}")


def compare_dataframes(df1, df2, name, detailed=False):
    """Compare two DataFrames."""
    print(f"\n{name}:")
    print("-" * 80)
    
    if df1 is None and df2 is None:
        print("  Both are None")
        return True
    elif df1 is None or df2 is None:
        print(f"  ✗ One is None: File1={df1 is not None}, File2={df2 is not None}")
        return False
    
    # Compare shapes
    if df1.shape != df2.shape:
        print(f"  ✗ Shape differs: {format_shape(df1.shape)} vs {format_shape(df2.shape)}")
    else:
        print(f"  ✓ Shape: {format_shape(df1.shape)}")
    
    # Compare columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"  Columns: {len(cols1)} vs {len(cols2)} ({len(common_cols)} common)")
    
    if only_in_1:
        print(f"  ✗ Only in File1: {sorted(only_in_1)}")
    if only_in_2:
        print(f"  ✗ Only in File2: {sorted(only_in_2)}")
    
    if not only_in_1 and not only_in_2:
        print(f"  ✓ All columns match")
    
    # Compare index
    if not df1.index.equals(df2.index):
        print(f"  ✗ Index differs (length: {len(df1.index)} vs {len(df2.index)})")
        if detailed and len(df1.index) == len(df2.index):
            diff_idx = df1.index != df2.index
            if diff_idx.any():
                n_diff = diff_idx.sum()
                print(f"    {n_diff} indices differ")
    else:
        print(f"  ✓ Index matches")
    
    # Compare values for common columns
    if detailed and common_cols:
        print("\n  Comparing values in common columns:")
        for col in sorted(common_cols):
            if col not in df1.columns or col not in df2.columns:
                continue
            try:
                if df1[col].dtype == 'object' or df2[col].dtype == 'object':
                    if not df1[col].equals(df2[col]):
                        n_diff = (df1[col] != df2[col]).sum()
                        print(f"    ✗ {col}: {n_diff} values differ")
                    else:
                        print(f"    ✓ {col}: identical")
                else:
                    # Numeric comparison
                    if not np.allclose(df1[col].values, df2[col].values, equal_nan=True):
                        diff_mask = ~np.isclose(df1[col].values, df2[col].values, equal_nan=True)
                        n_diff = diff_mask.sum()
                        if n_diff > 0:
                            max_diff = np.nanmax(np.abs(df1[col].values[diff_mask] - df2[col].values[diff_mask]))
                            print(f"    ✗ {col}: {n_diff} values differ (max diff: {max_diff:.6e})")
                    else:
                        print(f"    ✓ {col}: identical")
            except Exception as e:
                print(f"    ? {col}: Error comparing ({str(e)})")
    
    return len(only_in_1) == 0 and len(only_in_2) == 0


def compare_dict_structure(dict1, dict2, name):
    """Compare the structure of two dictionaries."""
    print(f"\n{name}:")
    print("-" * 80)
    
    if dict1 is None and dict2 is None:
        print("  Both are None")
        return
    elif dict1 is None or dict2 is None:
        print(f"  ✗ One is None: File1={dict1 is not None}, File2={dict2 is not None}")
        return
    
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"  Keys: {len(keys1)} vs {len(keys2)} ({len(common_keys)} common)")
    
    if only_in_1:
        print(f"  ✗ Only in File1: {sorted(only_in_1)}")
    if only_in_2:
        print(f"  ✗ Only in File2: {sorted(only_in_2)}")
    
    if not only_in_1 and not only_in_2:
        print(f"  ✓ All keys match")
    
    # Compare shapes/types of common keys
    if common_keys:
        print("\n  Common keys comparison:")
        for key in sorted(common_keys):
            val1 = dict1[key]
            val2 = dict2[key]
            
            type1 = type(val1).__name__
            type2 = type(val2).__name__
            
            if type1 != type2:
                print(f"    ✗ {key}: type differs ({type1} vs {type2})")
            elif hasattr(val1, 'shape') and hasattr(val2, 'shape'):
                if val1.shape != val2.shape:
                    print(f"    ✗ {key}: shape differs ({format_shape(val1.shape)} vs {format_shape(val2.shape)})")
                else:
                    print(f"    ✓ {key}: {type1} {format_shape(val1.shape)}")
            else:
                print(f"    ✓ {key}: {type1}")


def compare_layers(adata1, adata2, detailed=False):
    """Compare layers in two AnnData objects."""
    print("\nLAYERS:")
    print("-" * 80)
    
    layers1 = set(adata1.layers.keys()) if adata1.layers else set()
    layers2 = set(adata2.layers.keys()) if adata2.layers else set()
    
    if not layers1 and not layers2:
        print("  Both have no layers")
        return
    
    common_layers = layers1 & layers2
    only_in_1 = layers1 - layers2
    only_in_2 = layers2 - layers1
    
    print(f"  Layers: {len(layers1)} vs {len(layers2)} ({len(common_layers)} common)")
    
    if only_in_1:
        print(f"  ✗ Only in File1: {sorted(only_in_1)}")
    if only_in_2:
        print(f"  ✗ Only in File2: {sorted(only_in_2)}")
    
    if common_layers:
        print("\n  Common layers comparison:")
        for layer in sorted(common_layers):
            shape1 = adata1.layers[layer].shape
            shape2 = adata2.layers[layer].shape
            if shape1 != shape2:
                print(f"    ✗ {layer}: shape differs ({format_shape(shape1)} vs {format_shape(shape2)})")
            else:
                print(f"    ✓ {layer}: {format_shape(shape1)}")
                
                if detailed:
                    # Compare values
                    try:
                        if not np.allclose(adata1.layers[layer], adata2.layers[layer], equal_nan=True):
                            diff = np.abs(adata1.layers[layer] - adata2.layers[layer])
                            max_diff = np.nanmax(diff)
                            mean_diff = np.nanmean(diff)
                            print(f"      ✗ Values differ (max: {max_diff:.6e}, mean: {mean_diff:.6e})")
                        else:
                            print(f"      ✓ Values identical")
                    except Exception as e:
                        print(f"      ? Error comparing values: {str(e)}")


def compare_X(adata1, adata2, detailed=False):
    """Compare the main data matrix X."""
    print("\nMAIN DATA MATRIX (X):")
    print("-" * 80)
    
    if adata1.X is None and adata2.X is None:
        print("  Both are None")
        return
    elif adata1.X is None or adata2.X is None:
        print(f"  ✗ One is None: File1={adata1.X is not None}, File2={adata2.X is not None}")
        return
    
    shape1 = adata1.X.shape
    shape2 = adata2.X.shape
    
    if shape1 != shape2:
        print(f"  ✗ Shape differs: {format_shape(shape1)} vs {format_shape(shape2)}")
    else:
        print(f"  ✓ Shape: {format_shape(shape1)}")
    
    # Compare data type
    try:
        dtype1 = adata1.X.dtype
        dtype2 = adata2.X.dtype
        if dtype1 != dtype2:
            print(f"  ✗ dtype differs: {dtype1} vs {dtype2}")
        else:
            print(f"  ✓ dtype: {dtype1}")
    except:
        pass
    
    # Compare sparsity
    try:
        from scipy.sparse import issparse
        sparse1 = issparse(adata1.X)
        sparse2 = issparse(adata2.X)
        if sparse1 != sparse2:
            print(f"  ✗ Sparsity differs: File1={'sparse' if sparse1 else 'dense'}, File2={'sparse' if sparse2 else 'dense'}")
        else:
            print(f"  ✓ Both are {'sparse' if sparse1 else 'dense'}")
    except:
        pass
    
    if detailed and shape1 == shape2:
        print("\n  Comparing values:")
        try:
            # Convert to dense if sparse
            X1 = adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X
            X2 = adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X
            
            if not np.allclose(X1, X2, equal_nan=True):
                diff = np.abs(X1 - X2)
                max_diff = np.nanmax(diff)
                mean_diff = np.nanmean(diff)
                nonzero_diff = np.count_nonzero(diff > 1e-10)
                total = diff.size
                print(f"    ✗ Values differ:")
                print(f"      - {nonzero_diff}/{total} elements differ (>{1e-10})")
                print(f"      - Max difference: {max_diff:.6e}")
                print(f"      - Mean difference: {mean_diff:.6e}")
            else:
                print(f"    ✓ Values are identical")
        except Exception as e:
            print(f"    ? Error comparing values: {str(e)}")


def compare_h5ad_files(file1_path, file2_path, detailed=False):
    """
    Compare two h5ad files and print their differences.
    
    Args:
        file1_path: Path to first h5ad file
        file2_path: Path to second h5ad file
        detailed: If True, perform detailed value comparisons
    """
    print("\n" + "="*80)
    print("H5AD FILE COMPARISON")
    print("="*80)
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    # Check if files exist
    if not Path(file1_path).exists():
        print(f"\nError: File not found: {file1_path}")
        return False
    if not Path(file2_path).exists():
        print(f"\nError: File not found: {file2_path}")
        return False
    
    # Load files
    print("\nLoading files...")
    try:
        adata1 = ad.read_h5ad(file1_path)
        print(f"  ✓ Loaded File 1: {adata1.shape}")
    except Exception as e:
        print(f"  ✗ Error loading File 1: {str(e)}")
        return False
    
    try:
        adata2 = ad.read_h5ad(file2_path)
        print(f"  ✓ Loaded File 2: {adata2.shape}")
    except Exception as e:
        print(f"  ✗ Error loading File 2: {str(e)}")
        return False
    
    # Compare basic information
    compare_basic_info(adata1, adata2)
    
    # Compare main data matrix
    compare_X(adata1, adata2, detailed)
    
    # Compare obs (observations/cells metadata)
    print("\n" + "="*80)
    print("OBSERVATIONS (obs) METADATA")
    print("="*80)
    compare_dataframes(adata1.obs, adata2.obs, "obs", detailed)
    
    # Compare var (variables/genes metadata)
    print("\n" + "="*80)
    print("VARIABLES (var) METADATA")
    print("="*80)
    compare_dataframes(adata1.var, adata2.var, "var", detailed)
    
    # Compare layers
    print("\n" + "="*80)
    print("LAYERS")
    print("="*80)
    compare_layers(adata1, adata2, detailed)
    
    # Compare obsm (multi-dimensional observations)
    print("\n" + "="*80)
    print("MULTI-DIMENSIONAL OBSERVATIONS (obsm)")
    print("="*80)
    compare_dict_structure(adata1.obsm, adata2.obsm, "obsm")
    
    # Compare varm (multi-dimensional variables)
    print("\n" + "="*80)
    print("MULTI-DIMENSIONAL VARIABLES (varm)")
    print("="*80)
    compare_dict_structure(adata1.varm, adata2.varm, "varm")
    
    # Compare obsp (pairwise observations)
    print("\n" + "="*80)
    print("PAIRWISE OBSERVATIONS (obsp)")
    print("="*80)
    compare_dict_structure(adata1.obsp, adata2.obsp, "obsp")
    
    # Compare varp (pairwise variables)
    print("\n" + "="*80)
    print("PAIRWISE VARIABLES (varp)")
    print("="*80)
    compare_dict_structure(adata1.varp, adata2.varp, "varp")
    
    # Compare uns (unstructured annotations)
    print("\n" + "="*80)
    print("UNSTRUCTURED ANNOTATIONS (uns)")
    print("="*80)
    compare_dict_structure(adata1.uns, adata2.uns, "uns")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Compare two h5ad files and report their differences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_h5ad.py file1.h5ad file2.h5ad
  
  # Detailed comparison (includes value differences)
  python compare_h5ad.py file1.h5ad file2.h5ad --detailed
  
  # Save output to file
  python compare_h5ad.py file1.h5ad file2.h5ad > comparison_report.txt
        """
    )
    
    parser.add_argument('file1', help='Path to first h5ad file')
    parser.add_argument('file2', help='Path to second h5ad file')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Perform detailed value comparisons (slower but more comprehensive)')
    
    args = parser.parse_args()
    
    success = compare_h5ad_files(args.file1, args.file2, args.detailed)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

