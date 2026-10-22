'Download HEST-1k cancer subsets: breast IDC and kidney SCCRCC.'
import os, sys
from pathlib import Path

def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/hest')
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system(f'{sys.executable} -m pip install huggingface_hub --quiet')
        from huggingface_hub import hf_hub_download
    
    import pandas as pd
    
    # Download metadata
    meta_path = hf_hub_download(
        repo_id='MahmoodLab/hest', filename='HEST_v1_1_0.csv',
        repo_type='dataset', local_dir=str(base_dir),
    )
    meta = pd.read_csv(meta_path)
    print(f'Total HEST samples: {len(meta)}')
    
    # Check which column has the ID
    id_col = 'id' if 'id' in meta.columns else meta.columns[0]
    print(f'ID column: {id_col}')
    print(f'Technology column values: {meta["technology"].unique() if "technology" in meta.columns else "N/A"}')
    
    # Filter for Visium only
    if 'technology' in meta.columns:
        visium = meta[meta['technology'].str.contains('visium|Visium|ST', case=False, na=False)]
        print(f'Visium/ST samples: {len(visium)}')
    else:
        visium = meta
    
    # Download subsets
    subsets = {
        'breast_idc': visium[visium['oncotree_code'] == 'IDC'] if 'oncotree_code' in visium.columns else pd.DataFrame(),
        'kidney_sccrcc': visium[visium['oncotree_code'] == 'SCCRCC'] if 'oncotree_code' in visium.columns else pd.DataFrame(),
    }
    
    for name, subset in subsets.items():
        if len(subset) == 0:
            print(f'\n{name}: no samples found')
            continue
        
        out_dir = base_dir / name / 'st'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit to first 30 samples to manage space
        subset = subset.head(30)
        print(f'\n=== {name}: downloading {len(subset)} samples ===')
        
        success = 0
        for _, row in subset.iterrows():
            sample_id = row[id_col]
            fname = f'{sample_id}.h5ad'
            dest = out_dir / fname
            if dest.exists():
                print(f'  {fname}: exists')
                success += 1
                continue
            try:
                hf_hub_download(
                    repo_id='MahmoodLab/hest',
                    filename=f'st/{fname}',
                    repo_type='dataset',
                    local_dir=str(base_dir / name),
                )
                success += 1
                print(f'  {fname}: OK')
            except Exception as e:
                print(f'  {fname}: FAIL ({e})')
        
        print(f'  Downloaded: {success}/{len(subset)}')
    
    # Summary
    for name in subsets:
        d = base_dir / name / 'st'
        if d.exists():
            n = len(list(d.glob('*.h5ad')))
            print(f'{name}: {n} h5ad files')

if __name__ == '__main__':
    main()
