from sys import argv

def gen_topo(dp, tp, pp, dp_bw=5, tp_bw=10, pp_bw=5, dp_or_pp="dp", swap_dp_tp=False):
    # Node layout: node = pp_idx * (dp * tp) + dp_idx * tp + tp_idx
    total_npu = dp * tp * pp
    bw_matrix = [[0 for _ in range(total_npu)] for _ in range(total_npu)]
    for i in range(total_npu):
        i_pp = i // (dp * tp)
        i_dp = (i % (dp * tp)) // tp
        i_tp = i % tp
        for j in range(total_npu):
            j_pp = j // (dp * tp)
            j_dp = (j % (dp * tp)) // tp
            j_tp = j % tp

            # TP connections: same dp_idx and same pp_idx
            if i_dp == j_dp and i_pp == j_pp:
                bw_matrix[i][j] = tp_bw

            if dp_or_pp == "dp":
                # DP connections: same pp_idx, same tp_idx, adjacent dp_idx (ring with wrap-around)
                if i_pp == j_pp and i_tp == j_tp and (abs(i_dp - j_dp) == 1 or abs(i_dp - j_dp) == dp - 1):
                    bw_matrix[i][j] = dp_bw
            else:
                # PP connections: same dp_idx, same tp_idx, adjacent pp_idx (pipeline)
                if i_dp == j_dp and i_tp == j_tp and abs(i_pp - j_pp) == 1:
                    bw_matrix[i][j] = pp_bw

    if swap_dp_tp:
        # Swap DP and TP connections
        # build npu index mapping
        indices = [i for i in range(total_npu)]
        swapped_indices = []
        for i in range(total_npu):
            i_pp = i // (dp * tp)
            i_dp = (i % (dp * tp)) // tp
            i_tp = i % tp
            new_i = i_pp * (tp * dp) + i_tp * dp + i_dp
            swapped_indices.append(new_i)

        new_bw_matrix = [[0 for _ in range(total_npu)] for _ in range(total_npu)]
        for i in range(total_npu):
            for j in range(total_npu):
                new_i = swapped_indices[i]
                new_j = swapped_indices[j]
                new_bw_matrix[new_i][new_j] = bw_matrix[i][j]
        return new_bw_matrix

    return bw_matrix

def write_matrix_to_file(mat_dp, mat_pp, filename="schedules.txt", collapse_topo=False):
    with open(filename, "w") as f:
        f.write("BW 0\n")
        for row in mat_dp:
            f.write(" ".join(f"{x:3d}" for x in row) + "\n")
        f.write("END\n\n")
        if not collapse_topo:
            f.write("BW 1\n")
            for row in mat_pp:
                f.write(" ".join(f"{x:3d}" for x in row) + "\n")
            f.write("END\n")

if __name__ == "__main__":
    if len(argv) < 4:
        print("Usage: python topo_gen.py <dp> <tp> <pp> [<dp_bw>] [<tp_bw>] [<pp_bw>] [swap_dp_tp] [--collapse-topo]")
        exit(1)

    dp = int(argv[1])
    tp = int(argv[2])
    pp = int(argv[3])
    dp_bw = int(argv[4]) if len(argv) > 4 else 5
    tp_bw = int(argv[5]) if len(argv) > 5 else 10
    pp_bw = int(argv[6]) if len(argv) > 6 else 5
    swap_dp_tp = argv[7].lower() == "true" if len(argv) > 7 else False
    collapse_topo = "--collapse-topo" in argv

    mat_dp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, dp_or_pp="dp", swap_dp_tp=swap_dp_tp)
    mat_pp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, dp_or_pp="pp", swap_dp_tp=swap_dp_tp)

    # Collapse topo (Merge DP and PP into one matrix, selecting max bandwidth)
    if collapse_topo:
        mat_dp_pp = [[max(mat_dp[i][j], mat_pp[i][j]) for j in range(len(mat_dp))] for i in range(len(mat_dp))]
        write_matrix_to_file(mat_dp_pp, mat_dp_pp, collapse_topo=collapse_topo, filename="schedules-collapsed.txt")
    else:
        write_matrix_to_file(mat_dp, mat_pp, collapse_topo=collapse_topo)