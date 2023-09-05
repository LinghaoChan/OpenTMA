import numpy as np
import torch.nn.functional as F
from scipy.signal import normalize
import argparse
import os

def neg_recall(mat, k_value):
    neg_lists = []
    N = len(mat)
    for i in range(N):
        array = np.arange(N)
        np.random.shuffle(array)
        neg_list = list(array[:32])
        if i in neg_list:
            neg_list.remove(i)
        else:
            neg_list.pop()
        neg_lists.append(neg_list)
        # print(len(neg_list))
    hits = 0
    for rowid in range(len(mat)):
        row = mat[rowid]
        negsocres = list(row[neg_lists[rowid]])
        count_large = 0
        for one_score in negsocres:
            if row[rowid] < one_score:
                count_large += 1
        if count_large <= k_value - 1:
            hits += 1
    return hits

def main(args):
    # expdirs =  [
    #             "./results/temos/temos_humanml3d_kl_1e-5_wlatent/embeddings/test/epoch_0/"
    #             ]
    # retrieval_type = "T2M"
    # protocal = "D"
    expdirs = args.expdirs
    retrieval_type = args.retrieval_type
    protocal = args.protocal

    K_list = [1, 2, 3, 5, 10]
    RecK_list = [[] for i in expdirs]


    for index in range(len(expdirs)):
        exp_dir = expdirs[index]
        emb_dir = exp_dir
        # emb_dir = exp_dir + "/embeddings"
        motion_emb_dir = os.path.join(emb_dir, "motion_embedding.npy")
        text_emb_dir = os.path.join(emb_dir, "text_embedding.npy")
        sbert_emb_dir = os.path.join(emb_dir, "sbert_embedding.npy")

        text_embedding = np.load(text_emb_dir)
        motion_embedding = np.load(motion_emb_dir)
        sbert_embedding = np.load(sbert_emb_dir)

        sbert_embedding = sbert_embedding / np.linalg.norm(sbert_embedding, axis=1, keepdims=True)
        
        T2M_logits = text_embedding @ (motion_embedding.T)
        M2T_logits = motion_embedding @ (text_embedding.T)
        if retrieval_type == "T2M":
            logits_matrix = T2M_logits
        elif retrieval_type == "M2T":
            logits_matrix = M2T_logits

        sbert_sim = sbert_embedding @ (sbert_embedding.T)
        N = sbert_embedding.shape[0]

        target_list = []
        if protocal == "A" or protocal == "B":
            for i in range(N):
                target_list_i = []
                for j in range(N):
                    if protocal == "A":
                        if j==i:
                            target_list_i.append(j)
                    elif protocal == "B":
                        if sbert_sim[i][j] >= 0.9:
                            target_list_i.append(j)
                target_list.append(target_list_i)
            
            sorted_embedding_idx = np.argsort(-logits_matrix, axis=1)
            i = 0
            for k in K_list:
                hits = 0
                for i in range(N):
                    pred = list(sorted_embedding_idx[i][:k])
                    for item in pred:
                        if item in target_list[i]:
                            hits += 1
                            break
                # print(f'Recall @{k}: ', "%.3f" % (100.0 * (hits / N)), f'   hits={hits}', f'N={N}')
                RecK_list[index].append("%.3f" % (100.0 * (hits / N)))
                i += 1
                # print(f'Recall @{k}: | ', "%.3f" % (100.0 * (hits / N)))
        elif protocal == "D":
            for k in K_list:
                hits = neg_recall(logits_matrix, k)
                # print(f'Recall @{k}: | ', "%.3f" % (100.0 * (hits / N)))
                RecK_list[index].append("%.3f" % (100.0 * (hits / N)))
        # print(RecK_list)

    print('|   Metrics   |', end='  ')
    for k in K_list:
        print(f'Recall @{k} |', end='  ')
    print()
    print('|-------------|', end='  ')
    for k in K_list:
        print('--------- |', end='  ')
    print()
    for l in range(len(RecK_list)):
        exp_name = expdirs[l].split("/")[-2]
        print(f'|{exp_name} |', end='  ')
        for item in RecK_list[l]:
            print(item, end="   |")
        print("")
    print()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_type", default="T2M", type=str, help="T2M or M2T")
    parser.add_argument("--protocal", default="A", type=str, help="A, B, or D")
    parser.add_argument("--expdirs", nargs="+")
    args = parser.parse_args()
    main(args)