import os
import sys
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.Hobohm1 import cdhit_then_reassign

def check_first_line(file_path, keyword):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if keyword in first_line:
            return True
        else:
            return False


def MHC2pseudo(input_path, output_path, separator):
    input_data = pd.read_csv(input_path, sep=separator)
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
    #
    input_data_pseudo = input_data.copy()
    print("allele_mhc !")
    input_data_pseudo['mhc'] = input_data_pseudo['mhc'].replace(allele_mhc_dict)
    print("MHC_pseudo !")
    input_data_pseudo["new"] = input_data_pseudo['mhc'].apply(
        lambda x: '|'.join([MHC_pseudo_dict.get(i, i) for i in x.split(',')]))
    print("done !")
    input_data_pseudo['mhc'] = input_data_pseudo['new']
    input_data_pseudo.drop("new", axis=1, inplace=True)
    input_data_pseudo.to_csv(output_path, index=False)


def select_clusters_greedy(stratum, target_ma, target_sa, target_pos_ratio, rng, trials=100):
    """
    在层内随机搜索最佳簇划分子集，使测试集满足：
      - MA/SA 行数尽量接近 target_ma / target_sa
      - 正样本比例尽量接近 target_pos_ratio
    返回 (selected, remaining, cur_ma, cur_sa, cur_pos, cur_neg)
    """
    n = len(stratum)
    if n == 0:
        return [], [], 0, 0, 0, 0

    ma_arr = [c['ma_rows'] for c in stratum]
    sa_arr = [c['sa_rows'] for c in stratum]
    pos_arr = [c['pos_rows'] for c in stratum]
    neg_arr = [c['neg_rows'] for c in stratum]

    best_score = float('inf')
    best_selection = set()

    # 预计算行数平方和归一化因子，让分数无量纲
    norm = (target_ma + target_sa) ** 2 if (target_ma + target_sa) > 0 else 1.0

    for _ in range(trials):
        # 随机打乱簇顺序
        indices = list(range(n))
        rng.shuffle(indices)

        # 随机一个分割点，让测试集簇数在合理范围波动
        k = rng.randint(0, n)
        selected_idx = set(indices[:k])

        t_ma = sum(ma_arr[i] for i in selected_idx)
        t_sa = sum(sa_arr[i] for i in selected_idx)
        t_pos = sum(pos_arr[i] for i in selected_idx)
        t_neg = sum(neg_arr[i] for i in selected_idx)
        t_total = t_pos + t_neg
        t_ratio = t_pos / t_total if t_total > 0 else 0.0

        # 行数偏差 + 比例偏差（比例权重更高，确保正负对齐）
        score = ((t_ma - target_ma) ** 2 + (t_sa - target_sa) ** 2) / norm
        score += 10.0 * abs(t_ratio - target_pos_ratio)

        if score < best_score:
            best_score = score
            best_selection = selected_idx

    # 构建 selected / remaining 列表
    selected = [stratum[i] for i in range(n) if i in best_selection]
    remaining = [stratum[i] for i in range(n) if i not in best_selection]

    cur_ma = sum(ma_arr[i] for i in best_selection)
    cur_sa = sum(sa_arr[i] for i in best_selection)
    cur_pos = sum(pos_arr[i] for i in best_selection)
    cur_neg = sum(neg_arr[i] for i in best_selection)

    return selected, remaining, cur_ma, cur_sa, cur_pos, cur_neg


def random_split_n(n_splits=3, test_ratio=0.25, similarity_threshold=0.85, num_threads=50,
                   ma_test_size=710000, sa_test_size=350000):
    """
    1. 先拆出独立测试集（MA=ma_test_size, SA=sa_test_size），保证正负比例与总体一致
    2. 剩余数据（训练池）上做 n_splits 次拆分（train/valid），
       通过分层抽样保证每次拆分的正负比例一致
    所有拆分都保证同簇 epitope 不跨组，簇间相似度 < similarity_threshold。
    """
    df_el_SA_HLA = pd.read_csv("./data_random/train/df_el_SA_HLA.csv", header=0)
    df_el_MA_HLA = pd.read_csv("./data_random/train/df_el_MA_HLA.csv", header=0)
    all_epitopes = pd.concat([
        df_el_SA_HLA['epitope'],
        df_el_MA_HLA['epitope']
    ]).unique()

    print(f'\nUnique epitopes: {len(all_epitopes)}')
    print(f'Clustering with Hobohm1 (threshold={similarity_threshold})...')

    clusters = cdhit_then_reassign(list(all_epitopes), threshold=similarity_threshold,
                                   num_threads=num_threads)

    print(f'Clusters formed: {len(clusters)}')
    total_members = sum(len(v) for v in clusters.values())
    print(f"Total members in clusters: {total_members}  (should be {len(all_epitopes)})")

    # 计算每个 epitope 的 MA/SA 行数
    ma_epitope_counts = df_el_MA_HLA['epitope'].value_counts().to_dict()
    sa_epitope_counts = df_el_SA_HLA['epitope'].value_counts().to_dict()

    # 计算每个 epitope 的正/负样本行数（mass=1 为正，mass=0 为负）
    ma_pos_counts = df_el_MA_HLA[df_el_MA_HLA['mass'] == 1]['epitope'].value_counts().to_dict()
    ma_neg_counts = df_el_MA_HLA[df_el_MA_HLA['mass'] == 0]['epitope'].value_counts().to_dict()
    sa_pos_counts = df_el_SA_HLA[df_el_SA_HLA['mass'] == 1]['epitope'].value_counts().to_dict()
    sa_neg_counts = df_el_SA_HLA[df_el_SA_HLA['mass'] == 0]['epitope'].value_counts().to_dict()

    # 总体正负比例
    total_pos = sum(ma_pos_counts.values()) + sum(sa_pos_counts.values())
    total_neg = sum(ma_neg_counts.values()) + sum(sa_neg_counts.values())
    overall_pos_ratio = total_pos / (total_pos + total_neg)
    print(f'Overall: pos={total_pos:,}, neg={total_neg:,}, pos_ratio={overall_pos_ratio:.4f}')

    # 构建簇信息（含正负样本数）
    cluster_info = []
    for rep, members in clusters.items():
        members_set = set(members)
        ma_rows = sum(ma_epitope_counts.get(e, 0) for e in members_set)
        sa_rows = sum(sa_epitope_counts.get(e, 0) for e in members_set)
        pos_rows = (sum(ma_pos_counts.get(e, 0) for e in members_set) +
                    sum(sa_pos_counts.get(e, 0) for e in members_set))
        neg_rows = (sum(ma_neg_counts.get(e, 0) for e in members_set) +
                    sum(sa_neg_counts.get(e, 0) for e in members_set))
        cluster_info.append({
            'members': members_set,
            'ma_rows': ma_rows,
            'sa_rows': sa_rows,
            'pos_rows': pos_rows,
            'neg_rows': neg_rows,
        })

    output_dir = "./data_random/"
    os.makedirs(output_dir, exist_ok=True)

    # 按正负比例将簇分层
    def _pos_ratio(info):
        t = info['pos_rows'] + info['neg_rows']
        return info['pos_rows'] / t if t > 0 else 0

    n_strata = 4
    sorted_all = sorted(cluster_info, key=_pos_ratio)
    n_all = len(sorted_all)
    all_strata = []
    for s in range(n_strata):
        start = int(n_all * s / n_strata)
        end = int(n_all * (s + 1) / n_strata)
        all_strata.append(sorted_all[start:end])

    total_ma = sum(c['ma_rows'] for c in cluster_info)
    total_sa = sum(c['sa_rows'] for c in cluster_info)
    test_frac_ma = ma_test_size / total_ma
    test_frac_sa = sa_test_size / total_sa

    # ========== Step 1: 分层贪心抽取测试集 ==========
    print(f'\n=== Step 1: 分层测试集 (target MA={ma_test_size}, SA={sa_test_size}, '
          f'frac_ma={test_frac_ma:.3f}, frac_sa={test_frac_sa:.3f}) ===')
    test_epitopes = set()
    test_ma = test_sa = 0
    test_pos = test_neg = 0
    remaining_info = []

    for stratum in all_strata:
        st_ma = sum(c['ma_rows'] for c in stratum)
        st_sa = sum(c['sa_rows'] for c in stratum)
        target_ma = int(st_ma * test_frac_ma)
        target_sa = int(st_sa * test_frac_sa)

        # 在该层内贪心选择簇
        sel, rem, s_ma, s_sa, s_pos, s_neg = select_clusters_greedy(
            stratum, target_ma, target_sa, overall_pos_ratio, random.Random(42)
        )
        for info in sel:
            test_epitopes.update(info['members'])
        test_ma += s_ma
        test_sa += s_sa
        test_pos += s_pos
        test_neg += s_neg
        remaining_info.extend(rem)

    # 训练池 = 所有未入测试集的簇
    pool_epitopes = set()
    for info in remaining_info:
        pool_epitopes.update(info['members'])

    test_pos_ratio = test_pos / (test_pos + test_neg) if (test_pos + test_neg) > 0 else 0
    print(f'  Test:  {test_ma + test_sa:,} rows (MA={test_ma:,}, SA={test_sa:,}), '
          f'pos={test_pos:,}, neg={test_neg:,}, pos_ratio={test_pos_ratio:.4f}')

    # 保存测试集
    test_mask_ma = df_el_MA_HLA['epitope'].isin(test_epitopes)
    test_mask_sa = df_el_SA_HLA['epitope'].isin(test_epitopes)
    test_df = pd.concat([
        df_el_MA_HLA[test_mask_ma],
        df_el_SA_HLA[test_mask_sa]
    ], ignore_index=True)

    test_path = os.path.join(output_dir, 'df_el_test.csv')
    test_df.to_csv(test_path, index=False)
    print(f'  Saved to {test_path}')

    # ========== Step 2: 对训练池分层贪心抽取 valid ==========
    pool_total_ma = sum(c['ma_rows'] for c in remaining_info)
    pool_total_sa = sum(c['sa_rows'] for c in remaining_info)
    pool_pos = sum(c['pos_rows'] for c in remaining_info)
    pool_neg = sum(c['neg_rows'] for c in remaining_info)
    pool_pos_ratio = pool_pos / (pool_pos + pool_neg) if (pool_pos + pool_neg) > 0 else 0
    print(f'\n训练池: {pool_total_ma + pool_total_sa:,} rows (MA={pool_total_ma:,}, SA={pool_total_sa:,}), '
          f'pos_ratio={pool_pos_ratio:.4f}')
    print(f'=== Step 2: {n_splits} 次分层拆分 (train={1-test_ratio:.0%}, valid={test_ratio:.0%}) ===')

    # 对训练池重新分层
    sorted_pool = sorted(remaining_info, key=_pos_ratio)
    n_pool = len(sorted_pool)
    pool_strata = []
    for s in range(n_strata):
        start = int(n_pool * s / n_strata)
        end = int(n_pool * (s + 1) / n_strata)
        pool_strata.append(sorted_pool[start:end])

    for split_idx in range(n_splits):
        rng = random.Random(split_idx)

        valid_epitopes = set()
        valid_ma = valid_sa = 0
        valid_pos = valid_neg = 0
        train_epitopes = set()  # 直接构建 train 的 epitope 集合

        for stratum in pool_strata:
            st_ma = sum(c['ma_rows'] for c in stratum)
            st_sa = sum(c['sa_rows'] for c in stratum)
            target_ma = int(st_ma * test_ratio)
            target_sa = int(st_sa * test_ratio)

            sel, rem, s_ma, s_sa, s_pos, s_neg = select_clusters_greedy(
                stratum, target_ma, target_sa, overall_pos_ratio, rng
            )
            for info in sel:
                valid_epitopes.update(info['members'])
            valid_ma += s_ma
            valid_sa += s_sa
            valid_pos += s_pos
            valid_neg += s_neg

            # 未被选中的簇归入 train
            for info in rem:
                train_epitopes.update(info['members'])

        # 生成 dataframes
        valid_mask_ma = df_el_MA_HLA['epitope'].isin(valid_epitopes)
        valid_mask_sa = df_el_SA_HLA['epitope'].isin(valid_epitopes)
        train_mask_ma = df_el_MA_HLA['epitope'].isin(train_epitopes)
        train_mask_sa = df_el_SA_HLA['epitope'].isin(train_epitopes)

        valid_df = pd.concat([
            df_el_MA_HLA[valid_mask_ma],
            df_el_SA_HLA[valid_mask_sa]
        ], ignore_index=True)

        train_df = pd.concat([
            df_el_MA_HLA[train_mask_ma],
            df_el_SA_HLA[train_mask_sa]
        ], ignore_index=True)

        train_path = os.path.join(output_dir, f'df_el_train_split{split_idx}.csv')
        valid_path = os.path.join(output_dir, f'df_el_valid_split{split_idx}.csv')
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)

        # 统计 train/valid 的正负比例
        train_ma = pool_total_ma - valid_ma
        train_sa = pool_total_sa - valid_sa
        train_pos = pool_pos - valid_pos
        train_neg = pool_neg - valid_neg
        train_pr = train_pos / (train_pos + train_neg) if (train_pos + train_neg) > 0 else 0
        valid_pr = valid_pos / (valid_pos + valid_neg) if (valid_pos + valid_neg) > 0 else 0
        pct = (valid_ma + valid_sa) / (pool_total_ma + pool_total_sa) * 100
        print(f'\nSplit {split_idx}:')
        print(f'  Train: {train_ma + train_sa:,} rows (MA={train_ma:,}, SA={train_sa:,}), '
              f'pos_ratio={train_pr:.4f}')
        print(f'  Valid: {valid_ma + valid_sa:,} rows (MA={valid_ma:,}, SA={valid_sa:,}), '
              f'pos_ratio={valid_pr:.4f}  ({pct:.1f}%)')
        print(f'  Saved to {os.path.basename(train_path)} / {os.path.basename(valid_path)}')

    print(f'\nAll {n_splits} random splits saved to {output_dir}')


def seq_to_pseudo(filepath):
    outname = filepath.replace('.csv', '_pseudo.csv')
    MHC2pseudo(filepath, outname, ",")

def testdata_split():
    # 读取测试集数据
    data_path = './data_random/df_el_test.csv'
    allelelist_path = './data_random/allelelist'
    allele_mhc_dict = {}
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
    mask_ma = ~starts_hla
    df_sa = df[mask_sa].copy()
    df_ma = df[mask_ma].copy()

    df_sa.to_csv('./data_random/df_el_test_SA.csv', index=False)
    df_ma.to_csv('./data_random/df_el_test_MA.csv', index=False)
    
    # unique_mhc_counts = df.loc[mask_ma, 'mhc'].value_counts().sort_values(ascending=False)
    # # 取前20个
    # top_20_mhc = unique_mhc_counts.head(20).index.tolist()
    # os.makedirs(f"{output_dir}/MA_test/", exist_ok=True)

    # print("Unique MHC counts:")
    # print(unique_mhc_counts)

    
def get_SA_cluster_data():
    data_path = './data_random/SA_test/HLA_seen/'
    data_list = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        df_tmp = pd.read_csv(file_path, sep="\t", header=None, names=['epitope', 'mass', 'mhc'])
        data_list.append(df_tmp)
    data_all = pd.concat(data_list, ignore_index=True)
    data_all = data_all[data_all['mass'] == 1]
    print(len(data_all))

    data_all.to_csv('./data_random/SA_test/SA_HLA_seen_pos.txt', sep="\t", header=None)


if __name__ == '__main__':
    # 3次随机拆分（先拆独立测试集，再对训练池 80/20 随机拆）
    random_split_n(n_splits=3, test_ratio=0.25, similarity_threshold=0.85, num_threads=50,
                   ma_test_size=710000, sa_test_size=350000)

    seq_to_pseudo('./data_random/df_el_test.csv')
    for i in range(3):
        seq_to_pseudo(f'./data_random/df_el_train_split{i}.csv')
        seq_to_pseudo(f'./data_random/df_el_valid_split{i}.csv')

    testdata_split()
    # HLA_static()
    # get_SA_cluster_data()








