# Run ./run_network_reconfig.sh but edit the network.yml's reconfig_time according to input list
import os
import yaml
import subprocess


# find last occurence of (read from last lines)
# [2026-01-31 14:52:23.140] [workload] [info] sys[3] finished, 2369313572 cycles, exposed communication 2086471304 cycles.
# Extract cycles and exposed communication cycles
def extract_last_cycles(file_path):
    last_cycles = None
    last_exposed = None
    sys0_cycles = None
    sys0_exposed = None
    with open(file_path, 'r') as f:
        for line in f:
            if "finished" in line and "exposed communication" in line:
                # Only parse after "[WORKLOAD] [INFO]"
                line = line.split("[info]")[-1].strip()

                parts = line.split(',')
                cycles_part = parts[1].strip()
                exposed_part = parts[2].strip()

                cycles = int(cycles_part.split()[0])
                exposed_cycles = int(exposed_part.split()[2])

                last_cycles = cycles
                last_exposed = exposed_cycles

                if "sys[0] finished" in line:
                    sys0_cycles = cycles
                    sys0_exposed = exposed_cycles
                    
    return last_cycles, last_exposed, sys0_cycles, sys0_exposed

def run_with_reconfig_times(reconfig_times, base_dir, skip_analytical=False):
    results = {}

    if skip_analytical:
        print("Skipping analytical run")
        results['analytical'] = {
            'cycles': None,
            'exposed_cycles': None,
            'sys0_cycles': None,
            'sys0_exposed_cycles': None
        }
        results['baseline'] = {
            'cycles': None,
            'exposed_cycles': None,
            'sys0_cycles': None,
            'sys0_exposed_cycles': None
        }
    else:
        print(f"Running with fake analytical: ")
        subprocess.run(['bash', 'run_network_analytical.sh'], cwd=base_dir)
        output_analytical = os.path.join(base_dir, 'debug_analytical.txt')
        output_baseline = os.path.join(base_dir, 'debug_baseline.txt')
        cycles_analytical, exposed_analytical, sys0_cycles_analytical, sys0_exposed_analytical = extract_last_cycles(output_analytical)
        cycles_baseline, exposed_baseline, sys0_cycles_baseline, sys0_exposed_baseline = extract_last_cycles(output_baseline)

        results['analytical'] = {
            'cycles': cycles_analytical,
            'exposed_cycles': exposed_analytical,
            'sys0_cycles': sys0_cycles_analytical,
            'sys0_exposed_cycles': sys0_exposed_analytical
        }

        results['baseline'] = {
            'cycles': cycles_baseline,
            'exposed_cycles': exposed_baseline,
            'sys0_cycles': sys0_cycles_baseline,
            'sys0_exposed_cycles': sys0_exposed_baseline
        }

    for time in reconfig_times:
        # Load the existing network.yml
        network_file = os.path.join(base_dir, 'network.yml')


        with open(network_file, 'r') as f:
            network_config = yaml.safe_load(f)
        
        # Update the reconfig_time
        network_config['reconfig_time'] = [time * 1e9]  # Convert seconds to nanoseconds
        
        # Write back the updated network.yml
        with open(network_file, 'w') as f:
            yaml.dump(network_config, f)

        
        # Run the script
        print(f"Running with reconfig_time: {time}")
        subprocess.run(['bash', 'run_network_reconfig.sh'], cwd=base_dir)

        # Collect results from outputs (debug_no_provision.txt and debug_provision.txt)
        output_no_prov = os.path.join(base_dir, 'debug_no_provision.txt')
        output_prov = os.path.join(base_dir, 'debug_provision.txt')

        cycles_no_prov, exposed_no_prov, sys0_cycles_no_prov, sys0_exposed_no_prov = extract_last_cycles(output_no_prov)
        cycles_prov, exposed_prov, sys0_cycles_prov, sys0_exposed_prov = extract_last_cycles(output_prov)

        results[time] = {
            'no_provision': {
                'cycles': cycles_no_prov,
                'exposed_cycles': exposed_no_prov,
                'sys0_cycles': sys0_cycles_no_prov,
                'sys0_exposed_cycles': sys0_exposed_no_prov
            },
            'provision': {
                'cycles': cycles_prov,
                'exposed_cycles': exposed_prov,
                'sys0_cycles': sys0_cycles_prov,
                'sys0_exposed_cycles': sys0_exposed_prov
            }
        }
    
    return results

def write_result_for_sheet_import(results, filename):
    with open(filename, 'w') as f:
        f.write("Analytical\n")
        a_str = "\t".join([str(results['analytical']['sys0_cycles']), str(results['analytical']['cycles']), str(results['analytical']['exposed_cycles'])])
        f.write(a_str + "\n")
        f.write("\nBaseline\n")
        b_str = "\t".join([str(results['baseline']['sys0_cycles']), str(results['baseline']['cycles']), str(results['baseline']['exposed_cycles'])])
        f.write(b_str + "\n")
        f.write("\nReconfigurable\n")
        for time, data in results.items():
            if time == 'analytical':
                continue
            if time == 'baseline':
                continue
            np_str = "\t".join([str(data['no_provision']['sys0_cycles']), str(data['no_provision']['cycles']), str(data['no_provision']['exposed_cycles'])])
            time_str = f"{int(time)} s"
            if time < 1:
                time_str = f"{int(time*1000)} ms"
            
            f.write(f"{time_str}\t{np_str}\n")
            p_str = "\t".join([str(data['provision']['sys0_cycles']), str(data['provision']['cycles']), str(data['provision']['exposed_cycles'])])
            f.write(f"{time_str}\t{p_str}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reconfig experiments with different reconfig times")
    parser.add_argument("base_dir", help="Base directory containing the experiment files")
    parser.add_argument("--reconfig-times", default="0,0.01,0.05,0.1,0.25,0.5,0.75,1",
                        help="Comma-separated list of reconfig times in seconds (default: 0,0.01,0.05,0.1,0.25,0.5,0.75,1)")
    parser.add_argument("--skip-analytical", action="store_true",
                        help="Skip running the analytical/baseline experiments")
    args = parser.parse_args()

    base_dir = args.base_dir
    reconfig_times = [float(x) for x in args.reconfig_times.split(',')]

    results = run_with_reconfig_times(reconfig_times, base_dir, skip_analytical=args.skip_analytical)

    print("Results:")
    if not args.skip_analytical:
        print(f"Analytical - Cycles: {results['analytical']['cycles']}, Exposed Cycles: {results['analytical']['exposed_cycles']}")
        print(f"Baseline   - Cycles: {results['baseline']['cycles']}, Exposed Cycles: {results['baseline']['exposed_cycles']}")
    else:
        print("Analytical - Skipped")
        print("Baseline   - Skipped")

    for time, data in results.items():
        if time == 'analytical':
            continue
        if time == 'baseline':
            continue
        print(f"Reconfig Time: {time} seconds")
        print(f"  No Provision - Cycles: {data['no_provision']['cycles']}, Exposed Cycles: {data['no_provision']['exposed_cycles']}")
        print(f"  Provision    - Cycles: {data['provision']['cycles']}, Exposed Cycles: {data['provision']['exposed_cycles']}")

    write_result_for_sheet_import(results, os.path.join(base_dir, 'results_for_sheet_import.txt'))