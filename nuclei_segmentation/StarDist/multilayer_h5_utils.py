"""
Multi-layer H5 Utilities for Z-Stack Segmentation

This module provides utilities to save and load multi-layer segmentation data
with cell_id tracking for Active Learning consistency.

Author: AI Assistant
Date: 2025-10-22
"""

import h5py
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import time


def save_multilayer_segmentation(
    h5_path: str,
    cell_database: Dict[str, Dict],
    layer_data: Dict[int, Dict[str, np.ndarray]],
    embeddings: np.ndarray,
    reference_layer: int,
    layer_cell_ids: Optional[Dict[int, np.ndarray]] = None,
    reference_indices: Optional[np.ndarray] = None,
    node_name: str = "SegmentationNode"
) -> None:
    """
    Save multi-layer segmentation data to H5 file.
    
    Args:
        h5_path: Path to H5 file
        cell_database: Cell matching results (now from zstack_consistency.py)
        layer_data: {layer_idx: {'centroids': ..., 'contours': ..., 'probability': ...}}
        embeddings: [N, 768] fused embeddings for all unique cells
        reference_layer: Which layer is the reference
        layer_cell_ids: Pre-computed layer-to-cell_id mapping (from build_consistent_cell_database)
        reference_indices: Pre-computed reference indices (from build_consistent_cell_database)
        node_name: Group name in H5 (default: "SegmentationNode")
    """
    print(f"\n[H5 Multi-layer] Saving to: {h5_path}")
    print(f"  Reference layer: {reference_layer}")
    print(f"  Total layers: {len(layer_data)}")
    print(f"  Total cells: {len(cell_database)}")
    
    start_time = time.time()
    
    with h5py.File(h5_path, 'a') as hf:
        # Remove old data if exists
        if node_name in hf:
            del hf[node_name]
        
        seg_grp = hf.create_group(node_name)
        
        # 1. Save metadata
        meta_grp = seg_grp.create_group('metadata')
        meta_grp.attrs['is_multilayer'] = True 
        meta_grp.attrs['reference_layer'] = reference_layer
        meta_grp.attrs['total_layers'] = len(layer_data)
        meta_grp.attrs['total_cells'] = len(cell_database)
        meta_grp.attrs['matching_method'] = 'proximity'
        meta_grp.attrs['version'] = '2.0'  # Mark as multi-layer format
        
        print(f"[H5 Multi-layer] Saved metadata (is_multilayer=True)")
        
        # 2. Save cell database
        cell_grp = seg_grp.create_group('cell_database')
        
        # Sort cell_database by global_id for consistency
        sorted_cells = sorted(cell_database.items(), key=lambda x: x[0])
        
        # Cell IDs
        cell_ids = [cell_id for cell_id, _ in sorted_cells]
        cell_grp.create_dataset('cell_ids', data=np.array(cell_ids, dtype='S50'))
        
        # Reference indices
        if reference_indices is not None:
            cell_grp.create_dataset('reference_indices', data=reference_indices.astype(np.int32))
        else:
            # Fallback: use reference_idx from cell_database
            ref_indices = []
            for _, cell_info in sorted_cells:
                if 'reference_idx' in cell_info:
                    ref_indices.append(cell_info['reference_idx'])
                elif reference_layer in cell_info['layers']:
                    ref_indices.append(cell_info['layers'][reference_layer]['local_idx'])
                else:
                    ref_indices.append(-1) # No reference (cell not in reference layer)
            cell_grp.create_dataset('reference_indices', data=np.array(ref_indices, dtype=np.int32))
        
        # Embeddings (fused from all layers)
        cell_grp.create_dataset('embeddings', data=embeddings.astype(np.float32))
        
        # Layer presence matrix [N_cells, N_layers]
        n_layers = len(layer_data)
        layer_presence = np.zeros((len(cell_database), n_layers), dtype=bool)
        for i, (_, cell_info) in enumerate(sorted_cells):
            for layer_idx in cell_info['layers'].keys():
                layer_presence[i, layer_idx] = True
        cell_grp.create_dataset('layer_presence', data=layer_presence)
        
        print(f"[H5 Multi-layer] Saved cell_database:")
        print(f"  - cell_ids: {len(cell_ids)}")
        print(f"  - embeddings: {embeddings.shape}")
        print(f"  - layer_presence: {layer_presence.shape}")
        
        # 3. Save per-layer data
        for layer_idx, data in layer_data.items():
            layer_grp = seg_grp.create_group(f'layer_{layer_idx}')
            
            # Save segmentation results
            layer_grp.create_dataset('centroids', data=data['centroids'].astype(np.int32))
            layer_grp.create_dataset('contours', data=data['contours'].astype(np.int32))
            layer_grp.create_dataset('probability', data=data['probability'].astype(np.float32))
            
            # Use pre-computed layer_cell_ids if available, otherwise fallback to old method
            if layer_cell_ids is not None and layer_idx in layer_cell_ids:
                local_to_cell_id = layer_cell_ids[layer_idx]
            else:
                # Fallback: create mapping (old method)
                local_to_cell_id = create_local_to_cell_id_mapping(
                    cell_database, layer_idx, len(data['centroids'])
                )
            layer_grp.create_dataset('cell_ids', data=local_to_cell_id)
            
            print(f"[H5 Multi-layer] Saved layer_{layer_idx}:")
            print(f"  - centroids: {data['centroids'].shape}")
            print(f"  - contours: {data['contours'].shape}")
            print(f"  - cell_ids: {len(local_to_cell_id)}")
        
        # 4. BACKWARD COMPATIBILITY: Save reference layer data in old format location
        # This allows the old backend to read the data without modification
        print(f"\n[H5 Compat] Saving reference layer {reference_layer} to legacy format...")
        ref_data = layer_data[reference_layer]
        seg_grp.create_dataset('centroids', data=ref_data['centroids'].astype(np.int32))
        seg_grp.create_dataset('contours', data=ref_data['contours'].astype(np.int32))
        seg_grp.create_dataset('probability', data=ref_data['probability'].astype(np.float32))
        seg_grp.create_dataset('embedding', data=embeddings.astype(np.float32))
        print(f"[H5 Compat] Legacy format saved for backward compatibility")
        
        hf.flush()
    
    elapsed = time.time() - start_time
    print(f"[H5 Multi-layer] Save complete in {elapsed:.2f}s")


def create_local_to_cell_id_mapping(
    cell_database: Dict[str, Dict],
    layer_idx: int,
    n_local_cells: int
) -> np.ndarray:
    """
    Create mapping from local index to cell_id for a specific layer.
    
    Args:
        cell_database: Cell matching results
        layer_idx: Which layer
        n_local_cells: Number of cells in this layer's local arrays
    
    Returns:
        Array of cell_ids [n_local_cells], empty string if no match
    """
    local_to_cell = [''] * n_local_cells
    
    for cell_id, cell_info in cell_database.items():
        if layer_idx in cell_info['layers']:
            local_idx = cell_info['layers'][layer_idx]['local_idx']
            if 0 <= local_idx < n_local_cells:
                local_to_cell[local_idx] = cell_id
    
    return np.array(local_to_cell, dtype='S50')


def load_multilayer_segmentation(
    h5_path: str,
    node_name: str = "SegmentationNode"
) -> Dict[str, Any]:
    """
    Load multi-layer segmentation data from H5 file.
    
    Args:
        h5_path: Path to H5 file
        node_name: Group name in H5
    
    Returns:
        Dictionary with all multi-layer data
    """
    print(f"\n[H5 Multi-layer] Loading from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        if node_name not in hf:
            raise ValueError(f"Node '{node_name}' not found in H5 file")
        
        seg_grp = hf[node_name]
        
        # Check version
        if 'metadata' not in seg_grp:
            raise ValueError("Not a multi-layer format H5 file (no metadata group)")
        
        # Load metadata
        metadata = {
            'reference_layer': seg_grp['metadata'].attrs['reference_layer'],
            'total_layers': seg_grp['metadata'].attrs['total_layers'],
            'total_cells': seg_grp['metadata'].attrs['total_cells'],
            'matching_method': seg_grp['metadata'].attrs.get('matching_method', 'unknown'),
            'version': seg_grp['metadata'].attrs.get('version', '2.0')
        }
        
        print(f"[H5 Multi-layer] Metadata:")
        print(f"  Reference layer: {metadata['reference_layer']}")
        print(f"  Total layers: {metadata['total_layers']}")
        print(f"  Total cells: {metadata['total_cells']}")
        
        # Load cell database
        cell_db_grp = seg_grp['cell_database']
        cell_database = {
            'cell_ids': cell_db_grp['cell_ids'][:].astype(str),
            'reference_indices': cell_db_grp['reference_indices'][:],
            'embeddings': cell_db_grp['embeddings'][:],
            'layer_presence': cell_db_grp['layer_presence'][:]
        }
        
        # Load per-layer data
        layer_data = {}
        for layer_idx in range(metadata['total_layers']):
            layer_key = f'layer_{layer_idx}'
            if layer_key in seg_grp:
                layer_grp = seg_grp[layer_key]
                layer_data[layer_idx] = {
                    'centroids': layer_grp['centroids'][:],
                    'contours': layer_grp['contours'][:],
                    'probability': layer_grp['probability'][:],
                    'cell_ids': layer_grp['cell_ids'][:].astype(str)
                }
                print(f"[H5 Multi-layer] Loaded layer_{layer_idx}: {len(layer_data[layer_idx]['centroids'])} cells")
        
        result = {
            'metadata': metadata,
            'cell_database': cell_database,
            'layer_data': layer_data
        }
        
        print(f"[H5 Multi-layer] Load complete")
        
        return result


def get_legacy_format_data(
    h5_path: str,
    node_name: str = "SegmentationNode"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get data in legacy format (for backward compatibility).
    Returns data from reference layer.
    
    Args:
        h5_path: Path to H5 file
        node_name: Group name in H5
    
    Returns:
        (centroids, contours, embeddings) from reference layer
    """
    with h5py.File(h5_path, 'r') as hf:
        seg_grp = hf[node_name]
        
        # Check if multi-layer format
        if 'metadata' in seg_grp:
            # New format: return reference layer data
            ref_layer = seg_grp['metadata'].attrs['reference_layer']
            
            centroids = seg_grp[f'layer_{ref_layer}/centroids'][:]
            contours = seg_grp[f'layer_{ref_layer}/contours'][:]
            embeddings = seg_grp['cell_database/embeddings'][:]
            
            print(f"[Legacy Compat] Returning reference layer {ref_layer} data")
            return centroids, contours, embeddings
        else:
            # Old format: direct access
            centroids = seg_grp['centroids'][:]
            contours = seg_grp['contours'][:]
            embeddings = seg_grp['embedding'][:]
            
            print(f"[Legacy Compat] Returning old format data")
            return centroids, contours, embeddings


def get_layer_specific_data(
    h5_path: str,
    layer_idx: int,
    node_name: str = "SegmentationNode"
) -> Dict[str, np.ndarray]:
    """
    Get segmentation data for a specific layer.
    
    Args:
        h5_path: Path to H5 file
        layer_idx: Which layer to get
        node_name: Group name in H5
    
    Returns:
        Dictionary with centroids, contours, probability, cell_ids for that layer
    """
    with h5py.File(h5_path, 'r') as hf:
        seg_grp = hf[node_name]
        
        if 'metadata' not in seg_grp:
            raise ValueError("Not a multi-layer format H5 file")
        
        layer_key = f'layer_{layer_idx}'
        if layer_key not in seg_grp:
            raise ValueError(f"Layer {layer_idx} not found in H5 file")
        
        layer_grp = seg_grp[layer_key]
        
        return {
            'centroids': layer_grp['centroids'][:],
            'contours': layer_grp['contours'][:],
            'probability': layer_grp['probability'][:],
            'cell_ids': layer_grp['cell_ids'][:].astype(str)
        }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing multi-layer H5 utilities...\n")
    
    # Create synthetic data
    n_cells = 100
    reference_layer = 2
    
    # Create cell_database (from cell_matching.py output format)
    cell_database = {}
    for i in range(n_cells):
        cell_id = f"cell_{i:06d}"
        cell_database[cell_id] = {
            'cell_id': cell_id,
            'reference_layer': reference_layer,
            'reference_idx': i,
            'layers': {
                0: {'local_idx': i + 5, 'centroid': [100 + i, 200 + i]},
                1: {'local_idx': i + 3, 'centroid': [102 + i, 202 + i]},
                2: {'local_idx': i, 'centroid': [104 + i, 204 + i]},
                3: {'local_idx': i + 2, 'centroid': [106 + i, 206 + i]},
                4: {'local_idx': i + 4, 'centroid': [108 + i, 208 + i]},
            }
        }
    
    # Create layer_data
    layer_data = {}
    for layer_idx in range(5):
        n_local = n_cells + 10  # Some layers have more cells
        layer_data[layer_idx] = {
            'centroids': np.random.rand(n_local, 2) * 1000,
            'contours': np.random.rand(n_local, 32, 2) * 1000,
            'probability': np.random.rand(n_local)
        }
    
    # Create embeddings
    embeddings = np.random.rand(n_cells, 768).astype(np.float32)
    
    # Test save
    test_h5_path = "test_multilayer.h5"
    save_multilayer_segmentation(
        h5_path=test_h5_path,
        cell_database=cell_database,
        layer_data=layer_data,
        embeddings=embeddings,
        reference_layer=reference_layer
    )
    
    # Test load
    loaded_data = load_multilayer_segmentation(test_h5_path)
    
    print("\n[Test] Verifying loaded data...")
    assert loaded_data['metadata']['reference_layer'] == reference_layer
    assert loaded_data['metadata']['total_cells'] == n_cells
    assert len(loaded_data['cell_database']['cell_ids']) == n_cells
    assert loaded_data['cell_database']['embeddings'].shape == (n_cells, 768)
    
    # Test legacy format access
    centroids, contours, emb = get_legacy_format_data(test_h5_path)
    print(f"\n[Test] Legacy format data shapes:")
    print(f"  Centroids: {centroids.shape}")
    print(f"  Contours: {contours.shape}")
    print(f"  Embeddings: {emb.shape}")
    
    # Test layer-specific access
    layer_data_loaded = get_layer_specific_data(test_h5_path, layer_idx=2)
    print(f"\n[Test] Layer 2 specific data:")
    print(f"  Centroids: {layer_data_loaded['centroids'].shape}")
    print(f"  Cell IDs: {len(layer_data_loaded['cell_ids'])}")
    
    # Cleanup
    import os
    os.remove(test_h5_path)
    
    print("\n[OK] Multi-layer H5 utilities test complete!")

