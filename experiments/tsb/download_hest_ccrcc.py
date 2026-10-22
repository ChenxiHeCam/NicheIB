'Download HEST-1k ccRCC (clear cell renal cell carcinoma) subset.'
import os
import sys
from pathlib import Path

def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/hest_ccrcc')
    out_dir.mkdir(parents=True, exist_ok=True)
    st_dir = out_dir / 'st'
    st_dir.mkdir(exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        print('huggingface_hub available')
    except ImportError:
        print('Installing huggingface_hub...')
        os.system(f'{sys.executable} -m pip install huggingface_hub --quiet')
        from huggingface_hub import hf_hub_download, list_repo_files
    
    # First, download the metadata to find ccRCC samples
    print('Downloading HEST metadata...')
    meta_path = hf_hub_download(
        repo_id='MahmoodLab/hest',
        filename='HEST_v1_1_0.csv',
        repo_type='dataset',
        local_dir=str(out_dir),
    )
    
    import pandas as pd
    meta = pd.read_csv(meta_path)
    print(f'Total HEST samples: {len(meta)}')
    print(f'Columns: {list(meta.columns)[:20]}')
    
    # Filter for kidney / ccRCC
    # Check what organ/oncotree values are available
    if 'oncotree_code' in meta.columns:
        print(f'\nOncotree codes with counts > 5:')
        counts = meta['oncotree_code'].value_counts()
        print(counts[counts > 5])
    
    if 'organ' in meta.columns:
        print(f'\nOrgans:')
        print(meta['organ'].value_counts().head(20))
    
    # Filter for ccRCC
    ccrcc = meta[meta['oncotree_code'] == 'CCRCC'] if 'oncotree_code' in meta.columns else pd.DataFrame()
    print(f'\nccRCC samples: {len(ccrcc)}')
    
    if len(ccrcc) == 0:
        # Try kidney
        kidney = meta[meta['organ'].str.contains('kidney', case=False, na=False)] if 'organ' in meta.columns else pd.DataFrame()
        print(f'Kidney samples: {len(kidney)}')
        if len(kidney) > 0:
            print(kidney[['organ', 'oncotree_code', 'species', 'technology']].head(10) if all(c in kidney.columns for c in ['organ','oncotree_code','species','technology']) else kidney.head(10))
        
        # Also check breast, colorectal, lung
        for tissue in ['breast', 'colon', 'lung', 'kidney', 'pancrea']:
            subset = meta[meta['organ'].str.contains(tissue, case=False, na=False)] if 'organ' in meta.columns else pd.DataFrame()
            if len(subset) > 0:
                print(f'\n{tissue}: {len(subset)} samples')
                if 'oncotree_code' in subset.columns:
                    print(f'  Oncotree: {subset["oncotree_code"].value_counts().to_dict()}')
        return
    
    # Download h5ad files for ccRCC
    print(f'\nDownloading {len(ccrcc)} ccRCC h5ad files...')
    id_col = 'id' if 'id' in ccrcc.columns else ccrcc.columns[0]
    
    for idx, row in ccrcc.iterrows():
        sample_id = row[id_col]
        fname = f'{sample_id}.h5ad'
        dest = st_dir / fname
        if dest.exists():
            print(f'  {fname}: already exists, skipping')
            continue
        try:
            path = hf_hub_download(
                repo_id='MahmoodLab/hest',
                filename=f'st/{fname}',
                repo_type='dataset',
                local_dir=str(out_dir),
            )
            print(f'  {fname}: downloaded')
        except Exception as e:
            print(f'  {fname}: FAILED - {e}')
    
    # Count downloaded files
    downloaded = list(st_dir.glob('*.h5ad'))
    print(f'\nDownloaded {len(downloaded)} h5ad files to {st_dir}')

if __name__ == '__main__':
    main()
