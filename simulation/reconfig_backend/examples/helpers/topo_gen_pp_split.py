from sys import argv
import itertools

def gen_topo(dp, tp, pp, dp_bw=5, tp_bw=10, pp_bw=5, pp_groups=[], swap_dp_tp=False):
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

            # PP Connections
            if (abs(i_pp-j_pp) == 1 or abs(i_pp-j_pp) == pp - 1) and pp_groups[i_pp] and pp_groups[j_pp]:
                if i_dp == j_dp and i_tp == j_tp:
                    bw_matrix[i][j] = pp_bw

            # DP Connections
            if i_tp == j_tp and i_pp == j_pp:
                prev_pp = (i_pp + pp - 1) % pp
                next_pp = (i_pp + 1) % pp
                if not ((pp_groups[prev_pp] and pp_groups[i_pp]) or (pp_groups[next_pp] and pp_groups[i_pp])):
                    if abs(i_dp - j_dp) == 1 or abs(i_dp - j_dp) == dp - 1:
                        bw_matrix[i][j] = dp_bw

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

def write_matrix(id, bw_mat, filename="schedules.txt"):
    with open(filename, "a") as f:
        f.write(f"BW {id}\n")
        for row in bw_mat:
            f.write(" ".join(f"{x:3f}" for x in row) + "\n")
        f.write("END\n\n")

if __name__ == "__main__":
    if len(argv) != 9:
        print("python topo_gen.py <dp> <tp> <pp> <dp_bw> <tp_bw> <pp_bw> <swap_dp_tp> <mono-pp|analy|fg-pp>")
        exit(1)

    dp = int(argv[1])
    tp = int(argv[2])
    pp = int(argv[3])
    dp_bw = float(argv[4])
    tp_bw = float(argv[5])
    pp_bw = float(argv[6])
    swap_dp_tp = argv[7].lower() == "true"
    collapse_topo = "collapse-topo" in argv



    if "mono-pp" in argv:
        with open("schedules.txt", "w") as f:
            f.write("")
        pp_groups = [True for _ in range(pp)]  # Linear PP connections
        dp_groups = [False for _ in range(pp)]  # No PP connections
        mat_dp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, swap_dp_tp=swap_dp_tp, pp_groups=dp_groups)
        mat_pp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, swap_dp_tp=swap_dp_tp, pp_groups=pp_groups)
        write_matrix(0, mat_dp, filename="schedules.txt")
        write_matrix(1, mat_pp, filename="schedules.txt")

    elif "analy" in argv:
        with open("schedules-collapsed.txt", "w") as f:
            f.write("")
        pp_groups = [True for _ in range(pp)]  # Linear PP connections
        dp_groups = [False for _ in range(pp)]  # No PP connections
        mat_dp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, swap_dp_tp=swap_dp_tp, pp_groups=dp_groups)
        mat_pp = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, swap_dp_tp=swap_dp_tp, pp_groups=pp_groups)
        mat_dp_pp = [[max(mat_dp[i][j], mat_pp[i][j]) for j in range(len(mat_dp))] for i in range(len(mat_dp))]
        write_matrix(0, mat_dp_pp, filename="schedules-collapsed.txt")

    elif "fg-pp" in argv:
        with open("schedules.txt", "w") as f:
            f.write("")

        # Fine Grained PP
        pp_group_list = itertools.product([False, True], repeat=pp)

        for idx, sel in enumerate(pp_group_list):
            print(f"Generating FG-PP topology {idx} with PP connections: {sel}")
            mat = gen_topo(dp, tp, pp, dp_bw=dp_bw, tp_bw=tp_bw, pp_bw=pp_bw, swap_dp_tp=swap_dp_tp, pp_groups=sel)
            write_matrix(idx, mat, filename="schedules.txt")

    else: 
        print("No valid topology type specified. Use mono-pp, analy, or fg-pp.")
        exit(1)