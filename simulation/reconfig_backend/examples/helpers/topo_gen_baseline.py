from sys import argv
import itertools
import math

SCALE_OUT_BW = 100
SCALE_UP_BW = 450

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
            f.write(" ".join(f"{x:3f}" for x in row) + "\n")
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

def parse_link_stats_file(filename="link_stats_report.txt"):
    links = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Statistics Report" in line:
                continue
            if "Devices:" in line:
                parts = line.strip().split()
                total_npu = int(parts[2])
                links = [[0 for _ in range(total_npu)] for _ in range(total_npu)]
                continue
            if line.startswith("Link"):
                parts = line.strip().split()
                src = int(parts[1])
                dst = int(parts[3][:-1])
                bw = int(parts[8])
                links[src][dst] = bw
    return links

# Swapped DP and TP indices
# DP are closest to each other, then TP, then PP

# For example: DP=2 TP=2 PP=2
  
# ID DP TP PP
# 0   0  0  0
# 1   0  1  0
# 2   1  0  0
# 3   1  1  0 
# 4   0  0  1
# def get_dp_tp_pp_indices(npu_id, dp, tp, pp):
#     pp_idx = npu_id // (dp * tp)
#     dp_idx = (npu_id % (dp * tp)) // tp
#     tp_idx = npu_id % tp
#     return dp_idx, tp_idx, pp_idx

def calculate_total_bw(mat):
    total = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if i != j:
                total += mat[i][j]
    return total

def num_links_of_speed(mat, speed):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if i != j and mat[i][j] == speed:
                count += 1
    return count

def analyze_link_stats(links, dp, tp, pp):
    total_npu = dp * tp * pp
    dp_traffic = 0
    tp_traffic = 0
    pp_traffic = 0


    mat_pp = gen_topo(dp, tp, pp, 2, 3, 2, pp_groups=[True]*pp, swap_dp_tp=True)
    # write_matrix(1, mat_pp, filename="schedules_analysis.txt")
    pp_links = num_links_of_speed(mat_pp, 2)

    mat_dp = gen_topo(dp, tp, pp, 2, 3, 2, pp_groups=[False]*pp, swap_dp_tp=True)

    dp_links = num_links_of_speed(mat_dp, 2)
    tp_links = num_links_of_speed(mat_dp, 3)
    # write_matrix(0, mat_dp, filename="schedules_analysis.txt")

    print(f"Scale Out BW: {SCALE_OUT_BW}")
    print(f"Scale Up BW: {SCALE_UP_BW}")
    
    total_configured_bw = max(pp_links, dp_links) * SCALE_OUT_BW + tp_links * SCALE_UP_BW


    print(f"DP Links: {dp_links}, TP Links: {tp_links}, PP Links: {pp_links}")
    print(f"Total Configured BW: {total_configured_bw}")

    for i in range(total_npu):
        #i_dp, i_tp, i_pp = get_dp_tp_pp_indices(i, dp, tp, pp)
        for j in range(total_npu):
            #j_dp, j_tp, j_pp = get_dp_tp_pp_indices(j, dp, tp, pp)
            if i == j:
                continue

            # if links[i][j] == 0 and sample[i][j] != 0:
            #     print(f"Unused Link: {i} -> {j}, Configured BW: {sample[i][j]}")
            if mat_dp[i][j] == 2:
                dp_traffic += links[i][j]
                # print(f"DP Link: {i} -> {j}, Traffic: {links[i][j]}")
            elif mat_dp[i][j] == 3:
                tp_traffic += links[i][j]
            elif mat_pp[i][j] == 2:
                pp_traffic += links[i][j]


    print(f"Total DP Traffic: {dp_traffic}")
    print(f"Total TP Traffic: {tp_traffic}")
    print(f"Total PP Traffic: {pp_traffic}")

    return dp_traffic, tp_traffic, pp_traffic, dp_links, tp_links, pp_links, total_configured_bw

def allocate_bw(dp_traffic, tp_traffic, pp_traffic, dp_links, tp_links, pp_links, total_bw, simple=False):

    total_traffic = dp_traffic + tp_traffic + pp_traffic
    dp_total = total_bw * (dp_traffic / total_traffic)
    pp_total = total_bw * (pp_traffic / total_traffic)
    tp_total = total_bw * (tp_traffic / total_traffic)

    if not simple:
        factor = 1/(1+math.sqrt(dp_traffic/tp_traffic)+math.sqrt(pp_traffic/tp_traffic))
        tp_total = total_bw * factor
        dp_total = tp_total * math.sqrt(dp_traffic/tp_traffic)
        pp_total = tp_total * math.sqrt(pp_traffic/tp_traffic)

    dp_bw = dp_total / dp_links
    tp_bw = tp_total / tp_links
    pp_bw = pp_total / pp_links

    allocated_bw = dp_bw * dp_links + tp_bw * tp_links + pp_bw * pp_links
    print(f"Allocated BW: {allocated_bw}, Target BW: {total_bw}")
    return dp_bw, tp_bw, pp_bw

if __name__ == "__main__":

    if len(argv) != 6:
        print("python topo_gen_baseline.py <dp> <tp> <pp> <scale_out_bw> <scale_up_bw>")
        exit(1)

    dp = int(argv[1])
    tp = int(argv[2])
    pp = int(argv[3])
    SCALE_OUT_BW = float(argv[4])
    SCALE_UP_BW = float(argv[5])

    traffics = parse_link_stats_file(filename="link_stats_report.txt")
    dp_traffic, tp_traffic, pp_traffic, dp_links, tp_links, pp_links, total_bw = analyze_link_stats(traffics, dp, tp, pp)

    print(f"Total BW to allocate: {total_bw}")
    dp_bw, tp_bw, pp_bw = allocate_bw(dp_traffic, tp_traffic, pp_traffic, dp_links, tp_links, pp_links, total_bw)
    print(f"Allocated BW - DP: {dp_bw}, TP: {tp_bw}, PP: {pp_bw}")
    print(f"{round(tp_bw)} | {round(dp_bw)} | {round(pp_bw)} GB/s")

    dp_matrix = gen_topo(dp, tp, pp, dp_bw, tp_bw, pp_bw, pp_groups=[False]*pp, swap_dp_tp=True)
    pp_matrix = gen_topo(dp, tp, pp, dp_bw, tp_bw, pp_bw, pp_groups=[True]*pp, swap_dp_tp=True)

    baseline_matrix = [[max(dp_matrix[i][j], pp_matrix[i][j]) for j in range(len(dp_matrix))] for i in range(len(dp_matrix))]
    with open("schedules_baseline.txt", "w") as f:
        f.write("")
    write_matrix(0, baseline_matrix, filename="schedules_baseline.txt")