import gc
import os
import numpy as np
import pandas as pd
import torch
import esm
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_MIL import MHC_EL_split
from mhc_model_MIL import MHCpre_model_MIL_Capsule2
from torch.cuda.amp import autocast


PSEUDO_DAT = './data_random/MHC_pseudo.dat'

def hla_to_pseudo_key(allele):
    return allele.replace('*', '').replace(':', '')

def read_pseudo_dict():
    d = {}
    with open(PSEUDO_DAT) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d

pseudo_dict = read_pseudo_dict()

def df_to_pseudo(df):
    rows = []
    for _, row in df.iterrows():
        alleles = row['mhc'].split('|')
        pseudo_list = [pseudo_dict.get(hla_to_pseudo_key(a.strip()), a.strip())
                       for a in alleles if a.strip()]
        row = row.copy()
        row['mhc'] = '|'.join(pseudo_list)
        rows.append(row)
    return pd.DataFrame(rows)


def _ma_worker(gpu_id, file_names, temp_dir, model_dir, output_dir, input_dir):
    """Worker process: runs inference on assigned files across all 3 splits."""

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    print(f'[GPU {gpu_id}] Loading ESM-2 ...')
    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    for p in emb_model.parameters():
        p.requires_grad = False
    emb_model.to(device)
    emb_model.eval()

    results = {f: [] for f in file_names}
    features = {f: [] for f in file_names}
    attentions = {f: [] for f in file_names}

    for split_idx in [0, 1, 2]:
        print(f'[GPU {gpu_id}] Loading split {split_idx} checkpoint ...')
        model = MHCpre_model_MIL_Capsule2(emb_model)
        ckpt_path = os.path.join(model_dir, f'EL_Classification_train_split{split_idx}.ckpt')
        state_dict = torch.load(ckpt_path, map_location=device)
        state_dict = {k.replace('caps_net1.', 'caps_net.'): v for k, v in state_dict.items()}
        state_dict = {k.replace('fc.0.', 'fc_feat.0.'): v for k, v in state_dict.items()}
        state_dict = {k.replace('fc.3.', 'fc_out.1.'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        for fname in file_names:
            temp_path = os.path.join(temp_dir, fname.replace('.txt', '.csv'))
            testset = MHC_EL_split(temp_path, max_pep_len=15)
            testloader = DataLoader(testset, batch_size=128, shuffle=False,
                                    num_workers=2, pin_memory=True,
                                    collate_fn=testset.collate_fn)

            all_probs = []
            all_atts = []
            all_feats = []
            with torch.no_grad():
                for batch in tqdm(testloader, desc=f'GPU{gpu_id} s{split_idx} {fname}'):
                    input_data = batch['input_data'].to(device, non_blocking=True)
                    with autocast():
                        output, att_out, feat_out = model(input_data, batch['input_ids'])
                        probs = torch.sigmoid(output).cpu().numpy()
                        att_out = att_out.cpu().detach().tolist()
                        att_str = [",".join(str(x) for x in feat_vec) for feat_vec in att_out]
                        feat_str = feat_out
                    all_probs.append(probs)
                    all_atts += att_str
                    all_feats += feat_str
            results[fname].append(np.concatenate(all_probs).flatten())
            features[fname].append(all_feats)
            attentions[fname].append(all_atts)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    for fname in file_names:
        df = pd.read_csv(os.path.join(input_dir, fname), sep=' ', header=None, names=['epitope', 'mass'])
        df = df[['epitope']]
        for i, (res, feat, att) in enumerate(zip(results[fname], features[fname], attentions[fname])):
            df[f'prob_{i}'] = res
            df[f'attention_{i}'] = att
            df[f'feature_{i}'] = feat
        out_path = os.path.join(output_dir, fname)
        df.to_csv(out_path, sep=',', index=False)
        print(f'[GPU {gpu_id}] Saved {fname} ({len(df)} rows)')

    del emb_model
    torch.cuda.empty_cache()
    gc.collect()
    print(f'[GPU {gpu_id}] All done')


def data_predict(input_dir, output_dir, data_type="MA", gpu_ids=None):
    """
    Run MIL prediction on SA (single-allele) or MA (multi-allele) data.

    Args:
        input_dir: Directory containing input .txt files (one per allele/cell-line).
        output_dir: Directory to save predictions.
        data_type: "SA" for single-allele, "MA" for multi-allele.
        gpu_ids: List of GPU IDs to use. Defaults to [0] for SA, [0, 1, 3] for MA.
    """
    import shutil
    import multiprocessing as mp

    if gpu_ids is None:
        gpu_ids = [0] if data_type == "SA" else [0, 1, 3]

    pseudo_dat = './data_random/MHC_pseudo.dat'
    allelelist_path = './data_random/allelelist'
    model_dir = './model_MIL/'
    os.makedirs(output_dir, exist_ok=True)

    MHC_pseudo_dict = {}
    with open(pseudo_dat) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                MHC_pseudo_dict[parts[0]] = parts[1]

    allele_mhc_dict = {}
    with open(allelelist_path) as f:
        for line in f:
            try:
                allele_name, mhc_name = line.split("\t")
            except:
                print(line)
            allele_mhc_dict[allele_name.strip()] = mhc_name.strip()

    # External data: HLA allele groupings. Users should provide their own mapping file.
    hla_test_path = './data_random/HLA_test.txt'
    if os.path.exists(hla_test_path):
        with open(hla_test_path, "r") as f:
            for line in f.readlines():
                hla_name, hla_set = line.strip().split('\t', 1)
                hla_set_list = hla_set.strip().split('\t')
                hla_set_list = [item.replace('A', 'HLA-A').replace('B', 'HLA-B').replace('C', 'HLA-C') for item in hla_set_list]
                hla_list = [s[:-2] + ':' + s[-2:] for s in hla_set_list]
                allele_mhc_dict[hla_name] = ','.join(hla_list)

    # Prepare temporary pseudo CSV files
    if os.path.isdir(input_dir):
        file_list = sorted(os.listdir(input_dir))
    else:
        file_list = [os.path.basename(input_dir)]
        input_dir = os.path.dirname(input_dir)
    temp_dir = '/tmp/MA_MIL_predict/'
    os.makedirs(temp_dir, exist_ok=True)

    valid_files = []
    if data_type == "SA":
        for fname in file_list:
            if not fname.endswith('.txt'):
                continue
            hla_tmp = fname.replace('.txt', '')
            valid_files.append(fname)
            key = hla_tmp.replace(':', '').replace('*', '')
            pseudo_str = MHC_pseudo_dict.get(key, hla_tmp)

            df = pd.read_csv(os.path.join(input_dir, fname), sep='\t', header=None, names=['epitope', 'mass', 'mhc'])
            df_temp = df[['epitope', 'mass', 'mhc']].copy()
            df_temp['mhc'] = df_temp['mhc'].apply(lambda x: MHC_pseudo_dict[x])
            temp_path = os.path.join(temp_dir, fname.replace('.txt', '.csv'))
            df_temp.to_csv(temp_path, index=False)
    elif data_type == "MA":
        for fname in file_list:
            if not fname.endswith('.txt'):
                continue
            cell_line = fname.replace('.txt', '')
            if cell_line not in allele_mhc_dict:
                print(f'  Warning: {cell_line} not in allelelist, skipping')
                continue
            valid_files.append(fname)

            mhc_str = allele_mhc_dict[cell_line]
            mhc_list = mhc_str.split(",")
            pseudo_list = []
            for m in mhc_list:
                key = m.replace(':', '').replace('*', '')
                pseudo_list.append(MHC_pseudo_dict.get(key, m))
            pseudo_str = "|".join(pseudo_list)

            df = pd.read_csv(os.path.join(input_dir, fname), sep=' ', header=None, names=['epitope', 'mass'])
            df_temp = df[['epitope', 'mass']].copy()
            df_temp['mhc'] = pseudo_str
            temp_path = os.path.join(temp_dir, fname.replace('.txt', '.csv'))
            df_temp.to_csv(temp_path, index=False)

    # Parallel inference
    chunks = [[] for _ in gpu_ids]
    for i, fname in enumerate(valid_files):
        chunks[i % len(gpu_ids)].append(fname)

    print(f'\nDistributing {len(valid_files)} files across GPUs {gpu_ids}:')
    for gid, chunk in zip(gpu_ids, chunks):
        print(f'  GPU {gid}: {len(chunk)} files {chunk}')

    ctx = mp.get_context('spawn')
    processes = []
    for gid, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        p = ctx.Process(target=_ma_worker, args=(gid, chunk, temp_dir, model_dir, output_dir, input_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f'[WARNING] GPU worker exited with code {p.exitcode}')

    shutil.rmtree(temp_dir)
    print(f'\nAll done! Results in {output_dir}')


def predict_nature_SA(output_dir):
    """Predict on IEDB benchmark data (Nature SA benchmark)."""
    import shutil

    input_path = './data_random/IEDB_test.txt'
    pseudo_dat = './data_random/MHC_pseudo.dat'
    model_dir = './model_MIL/'
    temp_dir = '/tmp/SA_IEDB_predict/'
    DEVICE = torch.device('cuda:0')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    MHC_pseudo_dict = {}
    with open(pseudo_dat) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                MHC_pseudo_dict[parts[0]] = parts[1]

    df = pd.read_csv(input_path, sep='\t', header=0)
    print(f'Loaded {len(df)} rows from {input_path}')

    df['mhc'] = df['allele'].apply(
        lambda x: MHC_pseudo_dict.get(x.replace('*', '').replace(':', ''), x))
    df['mass'] = df['immunogenicity']
    df['epitope'] = df['peptide']
    temp_path = os.path.join(temp_dir, 'IEDB_pseudo.csv')
    df[['epitope', 'mass', 'mhc']].to_csv(temp_path, index=False)

    print('Loading ESM-2 ...')
    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    for p in emb_model.parameters():
        p.requires_grad = False
    emb_model.to(DEVICE)
    emb_model.eval()

    results = df[['epitope', 'allele', 'mass']].rename(columns={'allele': 'mhc', 'mass': 'label'}).copy()

    for split_idx in [0, 1, 2]:
        print(f'\n{"=" * 50}')
        print(f'  Inference for split {split_idx}')
        print(f'{"=" * 50}')

        model = MHCpre_model_MIL_Capsule2(emb_model)
        ckpt_path = os.path.join(model_dir, f'EL_Classification_train_split{split_idx}.ckpt')
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = {k.replace('caps_net1.', 'caps_net.'): v for k, v in state_dict.items()}
        state_dict = {k.replace('fc.0.', 'fc_feat.0.'): v for k, v in state_dict.items()}
        state_dict = {k.replace('fc.3.', 'fc_out.1.'): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model = model.to(DEVICE)
        model.eval()

        testset = MHC_EL_split(temp_path, max_pep_len=15)
        testloader = DataLoader(testset, batch_size=128, shuffle=False,
                                 num_workers=4, pin_memory=True,
                                 collate_fn=testset.collate_fn)

        all_probs = []
        with torch.no_grad():
            for batch in tqdm(testloader, desc=f'  split{split_idx}'):
                input_data = batch['input_data'].to(DEVICE)
                with autocast():
                    output, _, _ = model(input_data, batch['input_ids'])
                    probs = torch.sigmoid(output).float().cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs).flatten()
        results[f'prediction_{split_idx}'] = probs
        print(f'  Split {split_idx} done, {len(probs)} predictions')

        del model
        torch.cuda.empty_cache()
        gc.collect()

    result_path = os.path.join(output_dir, 'IEDB_BeitAI_predictions.csv')
    results.to_csv(result_path, index=False)
    print(f'\nSaved {result_path}: {len(results)} rows, columns: {list(results.columns)}')

    shutil.rmtree(temp_dir)
    print('All done!')


if __name__ == '__main__':
    input_dir = './data_random/MA_test/original/'
    output_dir = './output_MIL/MA_test/BeitAI-pHLA_result/'
    data_predict(input_dir, output_dir, data_type="MA")
