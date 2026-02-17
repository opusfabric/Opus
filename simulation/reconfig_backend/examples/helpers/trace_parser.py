from sys import argv
import yaml

def parse_tp_groups(file_path):
    with open(file_path, 'r') as f:
        tp_groups = []
        group_map = {}
        
        for line in f:
            """
            Mapped Comm Group 1 to Topology 0
            """
            # Groups that map to multiple topologies are TP groups
            if "Mapped Comm Group" in line:
                parts = line.split('to')
                group_part = parts[0].strip()
                topo_part = parts[1].strip()

                group_id = int(group_part.split()[3])
                topo_id = int(topo_part.split()[1])

                if group_id not in group_map:
                    group_map[group_id] = []
                group_map[group_id].append(topo_id)

        for group_id, topo_ids in group_map.items():
            if len(topo_ids) > 1:
                tp_groups.append(topo_ids)
            
    return tp_groups


def parse_comm_groups(file_path, tp_groups=[]):
    with open(file_path, 'r') as f:
        comm_idx_map = {}

        for line in f:
            """
            [1;33mScheduler: 3 current comm idx: 147, target comm group id: 2[0m
            """
            if "current comm idx" in line:
                parts = line.split(',')
                rank_part = parts[0].strip()
                comm_idx_part = parts[1].strip()
                group_id_part = parts[2].strip()

                rank = int(rank_part.split(':')[1].strip())
                current_comm_idx = int(comm_idx_part.split(':')[1].strip())
                target_group_id = int(group_id_part.split(':')[1].strip().replace('\x1b[0m', ''))

                if rank not in comm_idx_map:
                    comm_idx_map[rank] = {}
                if current_comm_idx != -1 and target_group_id != -2:
                    comm_idx_map[rank][current_comm_idx] = target_group_id
                    # if rank == 0:
                    #     print(f"[DEBUG] Rank {rank} parsed Comm Index {current_comm_idx} -> Group ID {target_group_id}")
    return comm_idx_map

def changed_comm_groups(comm_idx_map):
    changed_map = {}
    for rank, idx_map in comm_idx_map.items():
        changed_map[rank] = {}
        last_group_id = None
        last_idx = 0
        for idx in sorted(idx_map.keys()):
            group_id = idx_map[idx]
            # if rank == 0:
            #     print(f"[DEBUG] Rank {rank} checking Comm Index {idx} -> Group ID {group_id}")
            if group_id != last_group_id:
                #if last_group_id is not None:
                changed_map[rank][last_idx] = group_id
                # if rank == 0:
                #     print(f"[DEBUG] Rank {rank} changed Comm Index after {last_idx} -> Group ID {group_id}")
                last_group_id = group_id
            last_idx = idx
    return changed_map

def write_rank_comm_groups_yaml(changed_map, output_file):
    for rank in changed_map:
        new_map = {}
        for k, v in changed_map[rank].items():
            if v not in new_map:
                new_map[v] = []
            new_map[v].append(k)
        changed_map[rank] = new_map
        
    with open(output_file, 'w') as f:
        yaml.dump(changed_map, f)

if __name__ == "__main__":
    assert len(argv) == 2, "Usage: python trace_parser.py <trace_file_path>"
    file_path = argv[1]
    tp_groups = parse_tp_groups(file_path)
    comm_groups = parse_comm_groups(file_path, tp_groups)
    changed_groups = changed_comm_groups(comm_groups)
    # for rank, changes in changed_groups.items():
    #     print(f"Rank {rank}:")
    #     for idx, group_id in changes.items():
    #         print(f"  Comm Index {idx} -> Group ID {group_id}") 
    write_rank_comm_groups_yaml(changed_groups, 'rank_comm_groups.yaml')

