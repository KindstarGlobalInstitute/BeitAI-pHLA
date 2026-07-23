from collections import defaultdict

from tqdm import tqdm
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from multiprocessing import Pool
import os
import subprocess
import sys


def get_blocks(seq, k):
    """将序列切成 k 个尽量均匀的块（鸽巢分块）"""
    L = len(seq)
    q, r = divmod(L, k)
    blocks = []
    start = 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        blocks.append(seq[start:start+size])
        start += size
    return blocks

def _build_pigeonhole_index(reps, threshold):
    """建立鸽巢倒排索引：长度 -> 分块 -> 代表 ID 列表"""
    idx = defaultdict(lambda: defaultdict(list))
    for rep_id, seq in enumerate(reps):
        L = len(seq)
        d_max = int(L * (1 - threshold))   # 允许编辑距离
        k = d_max + 1
        if k < 1:
            k = 1
        blocks = get_blocks(seq, k)
        for block in blocks:
            idx[L][block].append(rep_id)
    return idx

def _query_worker(seq):
    """多进程 worker：在全局代表中寻找满足编辑距离的最佳代表 ID"""
    global _reps, _idx, _threshold
    L = len(seq)
    d_max = int(L * (1 - _threshold))
    k = d_max + 1 if d_max >= 0 else 1
    blocks = get_blocks(seq, k)

    # 收集候选
    candidates = set()
    for rep_len, block_dict in _idx.items():
        if min(L, rep_len) / max(L, rep_len) < _threshold:
            continue
        for block in blocks:
            for rep_id in block_dict.get(block, []):
                candidates.add(rep_id)

    best_rep = None
    # 按代表加入顺序（rep_id 升序）检查，模拟贪婪
    for rep_id in sorted(candidates):
        rep_seq = _reps[rep_id]
        if min(L, len(rep_seq)) / max(L, len(rep_seq)) < _threshold:
            continue
        d = Levenshtein.distance(seq, rep_seq)
        if d / max(L, len(rep_seq)) <= 1 - _threshold:
            best_rep = rep_id
            break
    return best_rep

def _init_worker(reps, idx, threshold):
    global _reps, _idx, _threshold
    _reps = reps
    _idx = idx
    _threshold = threshold

def cdhit_then_reassign(sequences, threshold=0.85, word_size=5, num_threads=140):
    """
    1. 运行 CD-HIT 获取代表序列（从 .clstr 文件提取）
    2. 用鸽巢索引 + 编辑距离将所有序列分配给代表
    返回 clusters 字典
    """
    # ---------- 写入输入 FASTA ----------
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f_in:
        input_file = f_in.name
        for i, seq in enumerate(sequences):
            f_in.write(f">seq_{i}\n{seq}\n")

    output_prefix = input_file + "_cdhit"
    clstr_file = output_prefix + ".clstr"

    cmd = [
        "cd-hit", "-i", input_file, "-o", output_prefix,
        "-c", str(threshold), "-n", str(word_size),
        "-d", "0", "-M", "0", "-T", str(num_threads), "-g", "1"
    ]

    print("[CD-HIT] 开始聚类，命令:", " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in process.stderr:
        sys.stderr.write(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

    # ---------- 从 .clstr 提取代表序列 ----------
    reps = []
    id_to_seq = {f"seq_{i}": seq for i, seq in enumerate(sequences)}
    with open(clstr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>') or not line:
                continue
            if '*' in line:   # 代表行
                # 提取 seq_XX
                after_gt = line.split('>', 1)[1]
                seq_id = after_gt.split('...')[0].split()[0]
                if seq_id in id_to_seq:
                    reps.append(id_to_seq[seq_id])

    print(f"从 .clstr 提取代表序列数: {len(reps)}")
    if len(reps) == 0:
        raise RuntimeError("未提取到任何代表序列，请检查 CD-HIT 输出文件")

    # ---------- 建立鸽巢索引 ----------
    print("建立鸽巢索引...")
    idx = _build_pigeonhole_index(reps, threshold)

    # ---------- 并行分配 ----------
    print(f"并行分配所有序列（{num_threads} 核）...")
    with Pool(processes=num_threads, initializer=_init_worker, initargs=(reps, idx, threshold)) as pool:
        results = list(tqdm(pool.imap(_query_worker, sequences, chunksize=1000),
                            total=len(sequences), desc='Reassigning'))

    # ---------- 构造 clusters ----------
    clusters = defaultdict(list)
    for seq, rep_id in zip(sequences, results):
        if rep_id is None:
            # 理论上不应发生，若发生则作为新代表
            rep_id = len(reps)
            reps.append(seq)
        clusters[reps[rep_id]].append(seq)

    # ---------- 清理临时文件 ----------
    try:
        os.remove(input_file)
        for ext in ('.clstr', '.bak.clstr', '.all.fa', '.rep.fa', '.rep'):
            fname = output_prefix + ext
            if os.path.exists(fname):
                os.remove(fname)
    except OSError:
        pass

    return dict(clusters)

