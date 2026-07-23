import os
import pandas as pd


def MHCstr_to_pseudo(MHC_name, allele_mhc_dict, MHC_pseudo_dict):
    mhc = allele_mhc_dict[MHC_name]
    mhc_list = mhc.split(",")
    pseudo_list = [MHC_pseudo_dict[m] for m in mhc_list]
    pseudo_str = "|".join(pseudo_list)
    return pseudo_str


def MA_data_split():
    # 读取测试集数据
    data_path = './data_random/df_el_test.csv'
    MHC_pseudo_dict = {}
    allele_mhc_dict = {}
    pseudo_path = './data_random/MHC_pseudo.dat'
    allelelist_path = './data_random/allelelist'
    with open(pseudo_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            MHC_name, pseudo_str = line.split(" ")
            MHC_pseudo_dict[MHC_name.strip()] = pseudo_str.strip()

    with open(allelelist_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                allele_name, mhc_name = line.split("\t")
            except:
                print(line)
            allele_mhc_dict[allele_name.strip()] = mhc_name.strip()
    
    
    df = pd.read_csv(data_path)
    output_dir = os.path.dirname(data_path)
    
    starts_hla = df['mhc'].str.startswith('HLA')
    has_comma = df['mhc'].str.contains(',')
    
    mask_sa = starts_hla & ~has_comma
    df_sa = df[mask_sa].copy()
    
    mask_ma = ~starts_hla
    unique_mhc_counts = df.loc[mask_ma, 'mhc'].value_counts().sort_values(ascending=False)
    # 取前20个
    top_20_mhc = unique_mhc_counts.head(20).index.tolist()
    os.makedirs(f"{output_dir}/MA_test/original/", exist_ok=True)
    os.makedirs(f"{output_dir}/MA_test/original_NoLabel/", exist_ok=True)
    os.makedirs(f"{output_dir}/MA_test/original_pseudo/", exist_ok=True)


    df = df[~df['epitope'].str.contains(r'[a-z]', na=False)]
    for mhc_str in top_20_mhc:
        df_subset = df[df['mhc'] == mhc_str]
        df_subset["pseudo"] = MHCstr_to_pseudo(mhc_str, allele_mhc_dict, MHC_pseudo_dict)
        df_subset['mhc'] = [allele_mhc_dict[x] for x in df_subset['mhc']]
        df_subset_1 = df_subset[['epitope', 'mass']]
        df_subset_2 = df_subset[['epitope']]


        df_subset_1.to_csv(f'{output_dir}/MA_test/original/{mhc_str}.txt', index=False, sep="\t", header=None)
        df_subset_2.to_csv(f'{output_dir}/MA_test/original_NoLabel/{mhc_str}.txt', index=False, sep="\t", header=None)

        # 删除mhc列，并将pseudo更名为mhc
        df_subset.drop(columns=['mhc'], inplace=True)
        df_subset.rename(columns={'pseudo': 'mhc'}, inplace=True)
        df_subset_3 = df_subset[['epitope', 'mass', 'mhc']]
        df_subset_3.to_csv(f'{output_dir}/MA_test/original_pseudo/{mhc_str}.txt', index=False, sep=",")
        print(mhc_str)



if __name__ == '__main__':
    MA_data_split()
