import sys

def filter_link_events(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            if "Got async error event" in line:
                if "port error" in line or "port active" in line:
                    print(line.strip())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python link_parser.py <file_path>")
    else:
        file_path = sys.argv[1]
        # filter_link_events(file_path)
        dev_activate_time_map = {}
        def parse_and_print_filtered_events(file_path):
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                for line in lines:
                    if "Got async error event" in line:
                        if "[rank0]" in line and ("port error" in line or "port active" in line) and "rocep13s0f1" in line:
                            parts = line.split()
                            time =  parts[1].split(']')[0]
                            time_in_second = float(time.split(':')[1])*60 + float(time.split(':')[2])
                            
                            device = parts[9].split(':')[0]
                            event = "error" if "port error" in line else "active"
                            print(f"Time: {time}, Device: {device}, Event: {event}")
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

        parse_and_print_filtered_events(file_path)