from sys import argv

def gen_topo(dp, tp, pp, dp_bw=5, tp_bw=10, pp_bw=5, dp_or_pp="dp"):
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

    return bw_matrix

def write_matrix_to_file(mat_dp, mat_pp, filename="schedules.txt"):
    with open(filename, "w") as f:
        f.write("BW 0\n")
        for row in mat_dp:
            f.write(" ".join(f"{x:3d}" for x in row) + "\n")
        f.write("END\n\n")
        f.write("BW 1\n")
        for row in mat_pp:
            f.write(" ".join(f"{x:3d}" for x in row) + "\n")
        f.write("END\n")

if __name__ == "__main__":
    if len(argv) < 4:
        print("Usage: python topo_gen.py <dp> <tp> <pp> [<dp_bw>] [<tp_bw>] [<pp_bw>]")
        exit(1)

    dp = int(argv[1])
    tp = int(argv[2])
    pp = int(argv[3])
    dp_bw = int(argv[4]) if len(argv) > 4 else 5
    tp_bw = int(argv[5]) if len(argv) > 5 else 10
    pp_bw = int(argv[6]) if len(argv) > 6 else 5

    mat_dp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, dp_or_pp="dp")
    mat_pp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, dp_or_pp="pp")

    write_matrix_to_file(mat_dp, mat_pp)