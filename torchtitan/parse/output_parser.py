import re
import sys

def extract_backend_comm_events(file_path):
    # Regular expression to match the required log lines
    pattern = re.compile(
        r'\[rank(\d+)\]:\x1b\[3\d+mBackend (\d+) RANK (\d+) .*? t:\[(\d+:\d+:\d+\.\d+)\]: Idx: (\d+), issuing (\w+), mode: \d+\x1b\[0m'
    )
    
    events = []
    
    with open(file_path, 'r') as file:
        latest_finish_time_in_ms = -1
        for line in file:
            match = pattern.search(line)

            if match:
                rank_id = int(match.group(1))
                backend_id = int(match.group(2))
                rank_id_in_group = int(match.group(3))
                time_parts = match.group(4).split(':')
                index = int(match.group(5))
                collective = match.group(6)

                time_in_ms = (int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])) * 1000

                if backend_id == 3:
                    continue

                # # 2D case, only read one iteration
                if rank_id != 0:
                    continue
                # if backend_id == 2 and index > 107:
                #     continue
                # if backend_id == 1 and index > 8:
                #     continue

                # Find the next line with "Collective finished!" for the same rank_id and backend_id
                next_line_pattern = re.compile(
                    rf'\[rank{rank_id}\]:\x1b\[3\d+mBackend {backend_id} RANK {rank_id_in_group} .*? t:\[(\d+:\d+:\d+\.\d+)\]: Collective finished!\x1b\[0m'
                )

                with open(file_path, 'r') as inner_file:
                    finish = False
                    for inner_line in inner_file:
                        next_match = next_line_pattern.search(inner_line)
                        if next_match:
                            finish_time_parts = next_match.group(1).split(':')
                            finish_time_in_ms = (
                                int(finish_time_parts[0]) * 3600 +
                                int(finish_time_parts[1]) * 60 +
                                float(finish_time_parts[2])
                            ) * 1000
                            if finish_time_in_ms > time_in_ms and (finish_time_in_ms > latest_finish_time_in_ms):
                                events.append({
                                    'rank': rank_id,
                                    'backend': backend_id,
                                    'rank_in_group': rank_id_in_group,
                                    'index': index,
                                    'time_ms': time_in_ms,
                                    'time': match.group(4),
                                    'finish_time': next_match.group(1),
                                    'duration_ms': finish_time_in_ms - time_in_ms,
                                    'coll': collective,
                                })
                                latest_finish_time_in_ms = finish_time_in_ms
                                finish = True
                                break

                    if not finish:
                        events.append({
                            'rank': rank_id,
                            'backend': backend_id,
                            'rank_in_group': rank_id_in_group,
                            'index': index,
                            'time_ms': time_in_ms,
                            'time': match.group(4),
                            'finish_time': 'n/a',
                            'duration_ms': 'n/a',
                            'coll': collective,
                        })
                        # print(f"Warning: No matching 'Collective finished!' found for rank {rank_id}, backend {backend_id}, index {index}")
                                

    events.sort(key=lambda event: event['time_ms'])
    
    return events

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python output_parser.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    events = extract_backend_comm_events(file_path)
    for event in events:
        # event.pop('time_ms')  # Remove time_ms before printing
        # event.pop('rank')
        event.pop('rank_in_group')
        event.pop('time_ms')
        if event['backend'] == 1:
            print("\033[94m", end="")  # Blue
        elif event['backend'] == 2:
            print("\033[92m", end="")  # Green
        elif event['backend'] == 3:
            print("\033[91m", end="")  # Red
        else:
            print("\033[0m", end="")  # Reset to default
        print(event)