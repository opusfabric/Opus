# Simple quick test script - creates & deletes some XCs and configures OPMs, OPM alarms and VOA if supported.
# Assumes that target switch (IP and port set in SUT.py) is a 48x48, and a VST is required to do anything with OPMs!

import time
import yaml
import socket
import signal
import sys
import os
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

IP_ADDRESS = os.environ.get('POLATIS_IP_ADDRESS', '10.253.51.12')
USERNAME = 'admin'
USERTYPE = 'admin'
PASSWORD = 'root'

# Parse command-line arguments for emulation mode, reconfig delay, and rail
parser = argparse.ArgumentParser(description="Configuration script for Opus controller.")
parser.add_argument("-e", action="store_true", help="Enable emulation mode")
parser.add_argument("reconfigDelay", type=int, help="Reconfiguration delay in milliseconds")
parser.add_argument("numRail", type=int, help="Number of rails")
parser.add_argument("rail", type=int, help="Rail ID")
args = parser.parse_args()

emulation_mode = args.e
reconfig_delay = args.reconfigDelay
rail = args.rail
num_rail = args.numRail

print(f"Starting Opus controller with emulation mode={'Enabled' if emulation_mode else 'Disabled'}, reconfig delay={reconfig_delay} ms, rail={rail}")

# Thread pool for emulation mode
NUM_TOPOLOGY_BITS = 64
thread_pool = None
topology_locks = {}
topology_states = {}
past_topo_lock = threading.Lock()  # Lock for past_topo variable

if emulation_mode:
    # Create thread pool with 64 workers (one per bit)
    thread_pool = ThreadPoolExecutor(max_workers=NUM_TOPOLOGY_BITS)
    # Initialize locks and states for each bit position
    for bit_pos in range(NUM_TOPOLOGY_BITS):
        topology_locks[bit_pos] = threading.Lock()
        topology_states[bit_pos] = 0

def emulate_reconfiguration(bit_pos, new_value, delay_ms):
    """Emulate reconfiguration for a specific bit position"""
    with topology_locks[bit_pos]:
        print(f"Rail {rail}: Thread {bit_pos} - Starting reconfiguration to {new_value}")
        time.sleep(delay_ms / 1000.0)
        topology_states[bit_pos] = new_value
        print(f"Rail {rail}: Thread {bit_pos} - Reconfiguration complete")

# get ProductInformation object
if not emulation_mode:
    from pypolatis import *
    # open session (USERNAME, USERTYPE, PASSWORD and IP_ADDRESS taken from SUT.py)
    session = Session(USERNAME, USERTYPE, IP_ADDRESS, timeout=30)
    session.login(PASSWORD)
    pi = session.productInformation()
    print('Info about SUT:')
    print(f'S/N: {pi.serialNumber()}')
    print(f'P/N: {pi.productCode()}')
    print(f'S/W: {pi.softwareVersion()}')

# Load configurations from a YAML file
config_file_path = os.environ.get(
    'OPUS_CONFIG_FILE',
    str(Path(__file__).resolve().parent / 'config' / 'config.yaml'),
)
with open(config_file_path, 'r') as file:
    configs = yaml.safe_load(file)["cross-connects"]

# Get CrossConnection object
if not emulation_mode:
    crossConnection = session.crossConnection()

# Define a signal handler to terminate the session gracefully
def signal_handler(sig, frame):
    print("Termination signal received. Logging out...")
    if thread_pool:
        thread_pool.shutdown(wait=True)
    if not emulation_mode:
        session.logout()
    print("Session terminated. Exiting.")
    sys.exit(0)

# Register the signal handler for termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

past_topo = -1

def handle_client_request(conn, addr):
    """Handle a single client connection in a separate thread"""
    global past_topo
    
    try:
        # Read configuration ID from IPC
        data = conn.recv(1024).decode().strip()
        if not data:
            conn.close()
            return
        
        if data.lower() == 'exit':
            print(f"Rail {rail}: Exiting...")
            if thread_pool:
                thread_pool.shutdown(wait=True)
            if not emulation_mode:
                session.logout()
            print(f"Rail {rail}: Session terminated. Exiting.")
            conn.close()
            sys.exit(0)
        
        config_id = int(data)
        
        if config_id < 0:
            conn.sendall(b"Invalid configuration ID\n")
            print(f"Rail {rail}: Invalid configuration ID received.")
            conn.close()
            return
        
        # Check if same as past topology (with lock)
        with past_topo_lock:
            if config_id == past_topo:
                conn.sendall(b"SUCCESS, CONFIG-NACK\n")
                print(f"Rail {rail}: Received same topology as previous, no changes made.")
                conn.close()
                return
            
            # Store the topology we're about to configure
            local_past_topo = past_topo
        
        # Apply selected configuration
        if emulation_mode:
            # Convert topology IDs to binary and find differences
            past_binary = format(local_past_topo if local_past_topo >= 0 else 0, f'0{NUM_TOPOLOGY_BITS}b')
            new_binary = format(config_id, f'0{NUM_TOPOLOGY_BITS}b')
            
            print(f"Rail {rail}: [{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] Applying topology {config_id} (binary: {new_binary})")
            
            # Submit tasks for changed bits
            futures = []
            for bit_pos in range(NUM_TOPOLOGY_BITS):
                if past_binary[bit_pos] != new_binary[bit_pos]:
                    new_value = int(new_binary[bit_pos])
                    future = thread_pool.submit(emulate_reconfiguration, bit_pos, new_value, reconfig_delay)
                    futures.append(future)
            
            # Wait for all reconfiguration threads to complete
            if futures:
                wait(futures)
                print(f"Rail {rail}: [{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] All reconfigurations complete")
            else:
                print(f"Rail {rail}: No bit changes detected")
        else:
            physical_topoId = 3 + (config_id - 1) * num_rail + rail
            selected_config = list(configs.keys())[physical_topoId]
            print(f"Rail {rail}: [{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] Applying {selected_config}...")
            crossConnection.setConnection(configs[selected_config]['ingress'], configs[selected_config]['egress'])
            time.sleep(reconfig_delay / 1000)  # Wait for the configuration to take effect
            print(f"Rail {rail}: [{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] Connections set")
            print(f"Rail {rail}: {crossConnection.connection()}")
        
        # Update past_topo after successful reconfiguration (with lock)
        with past_topo_lock:
            past_topo = config_id
        
        conn.sendall(b"SUCCESS, CONFIG-ACK\n")
        conn.close()
        
    except (ValueError, KeyError) as e:
        conn.sendall(b"Error processing configuration\n")
        print(f"Rail {rail}: Error: {e}")
        conn.close()
    except Exception as e:
        print(f"Rail {rail}: Unexpected error: {e}")
        conn.close()

# Set up IPC socket (only once, outside the loop)
ipc_dir = os.environ.get('OPUS_IPC_DIR', '/tmp')
ipc_prefix = os.environ.get('OPUS_IPC_PREFIX', 'opus_controller_ipc')
os.makedirs(ipc_dir, exist_ok=True)
IPC_SOCKET_PATH = os.path.join(ipc_dir, f"{ipc_prefix}_{rail}")
if len(IPC_SOCKET_PATH) >= 108:
    raise RuntimeError(f'IPC socket path is too long: {IPC_SOCKET_PATH}')
if os.path.exists(IPC_SOCKET_PATH):
    os.remove(IPC_SOCKET_PATH)

server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server_socket.bind(IPC_SOCKET_PATH)
server_socket.listen(5)  # Allow multiple pending connections
print(f"Rail {rail}: IPC server listening on {IPC_SOCKET_PATH}")

while True:
    conn, addr = server_socket.accept()
    # Handle each connection in a separate thread for non-blocking behavior
    handler_thread = threading.Thread(target=handle_client_request, args=(conn, addr))
    handler_thread.daemon = True
    handler_thread.start()
    