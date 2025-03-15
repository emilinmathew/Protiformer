import copy
import lmdb
import boto3
import os
import io
import threading
from multiprocessing import Value, Lock
import gzip
import torch
import pickle
import uuid
import hashlib
import time
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from transformers import AutoModel
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from prot2text.pdb2graph import PDB2Graph
from functools import partial
import tempfile
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor
from graphein.protein.features.nodes.dssp import phi, psi, asa, rsa, secondary_structure
from graphein.protein.edges.distance import (add_peptide_bonds,
                                           add_hydrogen_bond_interactions,
                                           add_disulfide_interactions,
                                           add_ionic_interactions,
                                           add_delaunay_triangulation,
                                           add_distance_threshold,
                                           add_sequence_distance_edges,
                                           add_k_nn_edges)


start_time = time.time()
processed_count = Value("i", 0)
count_lock = Lock()


# Constants
BUCKET_NAME = "cs-plm"
LMDB_S3_KEY = "embeddings/graph_embedding_runfeb26/data"
TEMP_LMDB_DIR = "/tmp/graph_embeddings_lmdb"
os.makedirs(TEMP_LMDB_DIR, exist_ok=True)
CHECKPOINT_FILE = "lmdb_keys.txt"

DOWNLOAD_CACHE_DIR = os.path.expanduser("~/pdb_cache")
GRAPH_CACHE_DIR = os.path.expanduser("~/graph_cache")
for cache_dir in [DOWNLOAD_CACHE_DIR, GRAPH_CACHE_DIR]:
    os.makedirs(cache_dir, exist_ok=True)
    for i in range(256):
        os.makedirs(os.path.join(cache_dir, f"{i:02x}"), exist_ok=True)

# l4 GPU specific
NUM_DOWNLOAD_WORKERS = 64  
NUM_PROCESSING_WORKERS = 16 
NUM_GPU_WORKERS = 2  
DOWNLOAD_BATCH_SIZE = 1000  
PROCESSING_BATCH_SIZE = 256  
GPU_BATCH_SIZE = 64 
S3_BATCH_SIZE = 3000  
LMDB_COMMIT_FREQUENCY = 10000  
CACHE_EXPIRY_DAYS = 7 
MAX_GPU_MEMORY = 20 * 1024 * 1024 * 1024 

_model = None
_graph_creator = None
_s3_client = None
_lmdb_env = None
_lmdb_writer_lock = threading.Lock()
_download_semaphore = threading.Semaphore(100)  
_s3_clients = {}  
_model_lock = threading.Lock()  

# Create protein graph config
config = {
    "node_metadata_functions": [
        amino_acid_one_hot,
        expasy_protein_scale,
        meiler_embedding,
        hydrogen_bond_acceptor, 
        hydrogen_bond_donor
    ],
    "edge_construction_functions": [
        add_peptide_bonds,
        add_hydrogen_bond_interactions,
        partial(add_distance_threshold, long_interaction_threshold=3, threshold=10.),
    ],
    "graph_metadata_functions": [
        asa, phi, psi, secondary_structure, rsa
    ],
    "dssp_config": DSSPConfig()
}
config = ProteinGraphConfig(**config)

def start_gpu_monitoring():
    def monitor_gpu():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"[GPU Monitor] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            time.sleep(5)
    
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    return monitor_thread
  
def get_lmdb_env():
    global _lmdb_env
    if _lmdb_env is None:
        _lmdb_env = lmdb.open(
            TEMP_LMDB_DIR,
            map_size=int(1e12),  
            writemap=True,  
            map_async=True,  
            max_dbs=1,      
            lock=True        
        )
    return _lmdb_env

def get_s3_client():
    thread_id = threading.get_ident()
    if thread_id not in _s3_clients:
        _s3_clients[thread_id] = boto3.client(
            "s3",
            config=boto3.session.Config(
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 10},
                max_pool_connections=100  
            )
        )
    return _s3_clients[thread_id]

def get_model():
    global _model
    with _model_lock:  
        if _model is None:
            device = "cuda:0"
            print("Using CUDA acceleration on T4 GPU")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True  
            
            _model = AutoModel.from_pretrained(
                "habdine/Prot2Text-Base-v1-1", 
                trust_remote_code=True, 
                force_download=False,
                local_files_only=False 
            )
            _model.eval() 
            
            # Apply mixed precision for T4
            _model = _model.half() 
            _model = _model.to(device)

            for name, param in _model.named_parameters():
                print(f"Parameter {name} is on {param.device}")
            
            if hasattr(torch, 'compile'):
                try:
                    _model = torch.compile(_model, mode="reduce-overhead")
                    print("Applied torch.compile optimization")
                except Exception as e:
                    print(f"Could not apply torch.compile: {e}")
                
    return _model

def estimate_batch_memory(batch_size, avg_node_count=500, avg_edge_count=2500):
    mem_per_graph = avg_node_count * avg_edge_count * 4 * 2  # Assuming 4 bytes per float, node+edge features
    mem_per_graph *= 2  
    return batch_size * mem_per_graph

def get_optimal_batch_size():
    try:
        free_memory, total_memory = torch.cuda.mem_get_info()
        available_memory = free_memory * 0.8  # Use 80% of available memory
        
        # Start with default batch size
        batch_size = GPU_BATCH_SIZE
        
        while estimate_batch_memory(batch_size) > available_memory and batch_size > 1:
            batch_size -= 4
        
        print(f"Dynamically set GPU batch size to {batch_size} based on available memory")
        return max(1, batch_size)
    except Exception as e:
        print(f"Error determining optimal batch size: {e}")
        return GPU_BATCH_SIZE  

def get_graph_creator():
    """Get or create PDB2Graph instance (reuse for efficiency)."""
    global _graph_creator
    if _graph_creator is None:
        _graph_creator = PDB2Graph(
            root=".",
            output_folder="graph_embeddings",
            config=config
        )
    return _graph_creator

def get_cache_path(pdb_key, cache_dir):
    """Get cache path with sharding for a PDB key."""
    filename = os.path.basename(pdb_key)
    hash_obj = hashlib.md5(pdb_key.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    shard = hash_hex[:2]
    
    return os.path.join(cache_dir, shard, f"{hash_hex}_{filename}")

def is_cache_valid(cache_path, max_age_days=CACHE_EXPIRY_DAYS):
    if not os.path.exists(cache_path):
        return False
    
    file_age_seconds = time.time() - os.path.getmtime(cache_path)
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    return file_age_seconds < max_age_seconds

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_checkpoint(processed_files):
    temp_file = f"{CHECKPOINT_FILE}.tmp"
    with open(temp_file, "w") as f:
        for file in processed_files:
            f.write(f"{file}\n")
    os.replace(temp_file, CHECKPOINT_FILE)  # Atomic replace

def list_pdb_files(processed_files=None, max_files=None):
    if processed_files is None:
        processed_files = load_checkpoint()
    
    s3 = get_s3_client()
    all_files = []
    continuation_token = None
    
    with tqdm(desc="Listing S3 files") as pbar:
        while True:
            params = {
                'Bucket': BUCKET_NAME,
                'MaxKeys': S3_BATCH_SIZE
            }
            if continuation_token:
                params['ContinuationToken'] = continuation_token
                
            response = s3.list_objects_v2(**params)

            if 'Contents' in response:
                batch_files = [
                    obj['Key'] for obj in response['Contents']
                    if obj['Key'].endswith('.pdb') and obj['Key'] not in processed_files
                ]
                all_files.extend(batch_files)
                pbar.update(len(batch_files))
                
                if max_files and len(all_files) >= max_files:
                    all_files = all_files[:max_files]
                    break
            
            if not response.get('IsTruncated'):
                break
                
            continuation_token = response.get('NextContinuationToken')
    
    print(f"Found {len(all_files)} PDB files to process")
    return all_files

def download_pdb_file(pdb_key):
    """Download a single PDB file with caching."""
    cache_path = get_cache_path(pdb_key, DOWNLOAD_CACHE_DIR)
    
    # Check cache first
    if is_cache_valid(cache_path):
        with open(cache_path, "r") as f:
            return pdb_key, cache_path, True  
    
    with _download_semaphore:
        try:
            s3 = get_s3_client()
            
            # Download file
            response = s3.get_object(Bucket=BUCKET_NAME, Key=pdb_key)
            raw_data = response["Body"].read()
            
            # i soemtimes gzipped it 
            if raw_data[:2] == b'\x1f\x8b':
                with gzip.GzipFile(fileobj=io.BytesIO(raw_data), mode="rb") as f:
                    pdb_data = f.read().decode("utf-8")
            else:
                pdb_data = raw_data.decode("utf-8")
                
            filtered_data = "\n".join(
                line for line in pdb_data.splitlines() if not line.startswith("DBREF")
            )
            
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, "w") as f:
                f.write(filtered_data)
            os.replace(temp_path, cache_path)  # Atomic replacement
                
            return pdb_key, cache_path, False  
            
        except Exception as e:
            print(f"Error downloading {pdb_key}: {e}")
            return pdb_key, None, False

def download_batch_parallel(pdb_keys):
    results = []
    
    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
        future_to_key = {executor.submit(download_pdb_file, key): key for key in pdb_keys}
        for future in tqdm(future_to_key, desc="Downloading PDBs", total=len(pdb_keys)):
            try:
                result = future.result()
                if result[1]:  # If path is not None
                    results.append(result)
            except Exception as e:
                key = future_to_key[future]
                print(f"Download failed for {key}: {e}")
    
    # Count cache hits
    cache_hits = sum(1 for _, _, from_cache in results if from_cache)
    print(f"Downloaded {len(results)} files ({cache_hits} from cache)")
    
    return results
  
def create_graph(pdb_key, pdb_path):
    graph_cache_path = get_cache_path(pdb_key, GRAPH_CACHE_DIR)

    # Check cache first
    if is_cache_valid(graph_cache_path):
        try:
            with open(graph_cache_path, 'rb') as f:
                graph = pickle.load(f)
                print(f"Loaded cached graph for {pdb_key}")
                return pdb_key, graph
        except Exception as e:
            print(f"Error loading cached graph for {pdb_key}: {e}")

    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated(0) / 1024**2
       
    # Not in cache, create graph with GPU acceleration
    try:
        graph_creator = get_graph_creator()
        graph = graph_creator.create_pyg_graph_gpu(pdb_path)

        # Verify graph device
        # if hasattr(graph, 'x'):
            # print(f"Graph is on device: {graph.x.device}")

        if torch.cuda.is_available():
          after_mem = torch.cuda.memory_allocated(0) / 1024**2
            # print(f"[After graph creation] Memory Allocated: {after_mem:.2f} MB")
          print(f"Memory delta: {after_mem - current_mem:.2f} MB")

        # Move to CPU before caching
        if graph.x.device.type == 'cuda':  
            # print(f"Moving graph from {graph.x.device} to CPU for caching")
            graph_cpu = copy.deepcopy(graph)
            for attr in graph_cpu.__dict__:
                value = getattr(graph_cpu, attr)
                if isinstance(value, torch.Tensor):
                    setattr(graph_cpu, attr, value.cpu())
        else:
            graph_cpu = graph
          
        lmdb_env = get_lmdb_env()
        # Save to LMDB for caching
        with lmdb_env.begin(write=True) as txn:
            txn.put(pdb_key.encode(), pickle.dumps(graph_cpu))

        return pdb_key, graph  # Graph remains on GPU for processing

    except Exception as e:
        print(f"Error creating graph for {pdb_key}: {e}")
        import traceback
        traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clean up GPU memory

        return pdb_key, None


def create_graphs_batch(batch_data):
    """Create graphs for a batch of PDB files with GPU acceleration."""
    results = {}
    
    # Force CUDA synchronization before processing plszzz
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"Starting batch with {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB GPU memory used")
    
    # Process in GPU-friendly batches
    optimal_batch_size = 8 
    
    for i in range(0, len(batch_data), optimal_batch_size):
        sub_batch = batch_data[i:i+optimal_batch_size]
        print(f"Processing sub-batch {i//optimal_batch_size+1} with {len(sub_batch)} PDB files")
        
        for key, path, _ in sub_batch:
            try:
                key, graph = create_graph(key, path)
                if graph is not None:
                    # Verify graph is on GPU
                    # if hasattr(graph, 'x'):
                        # print(f"Graph {key} is on {graph.x.device}")
                    results[key] = graph
            except Exception as e:
                print(f"Error processing {key}: {e}")
        
        # Force synchronization and clean GPU memory between batches
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            before_clean = torch.cuda.memory_allocated(0) / 1024**2
            torch.cuda.empty_cache()
            after_clean = torch.cuda.memory_allocated(0) / 1024**2
            print(f"Cleaned {before_clean - after_clean:.2f} MB from GPU")
            print(f"Batch {i//optimal_batch_size+1}: GPU memory: {after_clean:.2f} MB")
    
    return results
  
  

def process_gpu_batch(graph_batch):
    global processed_count
    try:
        # Get GPU model
        model = get_model()
        device = model.device
        print(f"Processing batch on {device} with {len(graph_batch)} graphs")
        
        # Track successful embeddings
        embeddings_batch = {}
        processed_keys = []
        
        # Process each graph
        for pdb_key, graph in graph_batch.items():
            batch_start = time.time()
            try:
                # Ensure graph is on correct device
                if not hasattr(graph, 'x') or graph.x is None:
                    print(f"Warning: Graph {pdb_key} has no features (x is None)")
                    continue
                    
                # Explicitly move to GPU
                if graph.x.device != device:
                    # print(f"Moving graph {pdb_key} from {graph.x.device} to {device}")
                    graph = graph.to(device)
                
                # Ensure inputs are float16 for mixed precision
                graph.x = graph.x.half()
                
                # Process edge types - check if they exist first
                if hasattr(graph, 'edge_type') and graph.edge_type is not None:
                    if len(graph.edge_type.shape) > 1 and graph.edge_type.shape[1] > 1:
                        edge_type_tensor = torch.tensor(graph.edge_type, dtype=torch.float32, device=device)
                        edge_type_labels = torch.argmax(edge_type_tensor, dim=1)
                        graph.edge_type = edge_type_labels
                    else:
                        print(f"Warning: edge_type for {pdb_key} has unexpected shape: {graph.edge_type.shape}")
                else:
                    print(f"Warning: Graph {pdb_key} has no edge_type")
                    continue
                
                if torch.cuda.is_available():
                    print(f"[Before inference] GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                
                # Extract embeddings with mixed precision
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        embeddings = model.encoder(graph.x, graph.edge_index, graph.edge_type, graph.batch)
                        graph_embedding = embeddings.squeeze(1)
                
                # Print memory usage after processing
                if torch.cuda.is_available():
                    print(f"[After inference] GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                
                with count_lock:
                  processed_count.value += 1  

                # Store embeddings
                embedding_np = graph_embedding.cpu().numpy()
                key = os.path.basename(pdb_key).encode('utf-8')
                embeddings_batch[key] = embedding_np
                processed_keys.append(pdb_key)
                
                # Clear GPU memory
                del graph, graph_embedding, embeddings
                torch.cuda.empty_cache()
              
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                avg_time_per_file = total_time / processed_count.value
                estimated_time_remaining = avg_time_per_file * (230000 - processed_count.value)
        
                # print(f"Processed {processed_count.value}/{230000} | Time per file: {batch_time:.2f}s | Estimated Time Left: {estimated_time_remaining / 3600:.2f} hours")

            except Exception as e:
                print(f"Error processing graph {pdb_key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Batch write to LMDB
        if embeddings_batch:
            lmdb_env = get_lmdb_env()
            with _lmdb_writer_lock:
                with lmdb_env.begin(write=True) as txn:
                    for key, value in embeddings_batch.items():
                        txn.put(key, pickle.dumps(value))
        
        # Clear any remaining GPU memory
        torch.cuda.empty_cache()
        
        return processed_keys
    except Exception as e:
        print(f"Error in GPU batch processing: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()  # Ensure GPU memory is cleared even on error
        return []
      

def dynamic_gpu_batching(graphs, max_batch_size=None):
    """Dynamically batch graphs for GPU processing based on GPU memory."""
    if max_batch_size is None:
        max_batch_size = get_optimal_batch_size()
    
    batches = []
    current_batch = {}
    current_size = 0
    
    # Sort graphs by estimated size (node count) to better pack batches
    sorted_items = sorted(
        graphs.items(), 
        key=lambda item: (len(item[1].x) if hasattr(item[1], 'x') else 0), 
        reverse=True  # Process largest graphs first
    )
    
    for key, graph in sorted_items:
        # Estimate graph size
        graph_size = 1  # Base size unit
        
        # Add to current batch if it fits
        if current_size + graph_size <= max_batch_size:
            current_batch[key] = graph
            current_size += graph_size
        else:
            # Start a new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = {key: graph}
            current_size = graph_size
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def process_pdbs_with_pipeline(file_keys, batch_size=PROCESSING_BATCH_SIZE):
    """Process PDBs with an optimized pipeline for T4 GPU."""
    processed_files = load_checkpoint()
    remaining_files = [f for f in file_keys if f not in processed_files]
    total_files = len(remaining_files)
    
    print(f"Processing {total_files} files with T4-optimized pipeline")
    start_time = time.time()
    
    # Process in large batches to benefit from parallelism
    pbar = tqdm(total=total_files, desc="Overall Progress")
    commit_counter = 0
    
    for i in range(0, len(remaining_files), DOWNLOAD_BATCH_SIZE):
        # Step 1: Download a large batch of files in parallel
        batch_files = remaining_files[i:i + DOWNLOAD_BATCH_SIZE]
        downloaded_batch = download_batch_parallel(batch_files)
        
        # Step 2: Create graphs in parallel (CPU-bound task)
        graphs = create_graphs_batch(downloaded_batch)
        
        # Step 3: Process graphs in optimized GPU batches
        gpu_batches = dynamic_gpu_batching(graphs)
        print(f"Created {len(gpu_batches)} GPU batches from {len(graphs)} graphs")
        
        newly_processed = []
        for batch_idx, graph_batch in enumerate(gpu_batches):
            batch_start = time.time()
            batch_results = process_gpu_batch(graph_batch)
            batch_time = time.time() - batch_start
            
            newly_processed.extend(batch_results)
            processed_files.update(batch_results)
            commit_counter += len(batch_results)
            pbar.update(len(batch_results))
            
            # Print batch stats
            if batch_results:
                items_per_sec = len(batch_results) / batch_time
                print(f"Batch {batch_idx+1}/{len(gpu_batches)}: Processed {len(batch_results)} items in {batch_time:.2f}s ({items_per_sec:.2f} items/sec)")
            
            # Checkpoint periodically
            if commit_counter >= LMDB_COMMIT_FREQUENCY:
                save_checkpoint(processed_files)
                elapsed = time.time() - start_time
                files_per_sec = len(processed_files) / elapsed
                estimated_total = elapsed * (total_files / max(1, len(processed_files)))
            
                # Clear memory and reset counter
                torch.cuda.empty_cache()
                commit_counter = 0
        
        # Clear memory between large batches
        del graphs
        torch.cuda.empty_cache()
    
    # Final checkpoint
    if commit_counter > 0:
        save_checkpoint(processed_files)
    
    pbar.close()
    
    total_time = time.time() - start_time
  
    return processed_files

def clean_cache():
    """Clean expired cache entries to free up disk space."""
    cache_dirs = [DOWNLOAD_CACHE_DIR, GRAPH_CACHE_DIR]
    files_removed = 0
    bytes_freed = 0
    
    for cache_dir in cache_dirs:
        print(f"Cleaning cache in {cache_dir}...")
        for root, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if not is_cache_valid(file_path):
                    size = os.path.getsize(file_path)
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        bytes_freed += size
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
    
    print(f"Removed {files_removed} expired cache files ({bytes_freed / (1024*1024):.2f} MB)")

def upload_lmdb_to_s3():
    """Upload LMDB to S3 with multipart upload for better performance."""
    s3 = get_s3_client()
    
    # Close the environment to ensure all data is flushed
    global _lmdb_env
    if _lmdb_env is not None:
        _lmdb_env.close()
        _lmdb_env = None
    
    data_path = os.path.join(TEMP_LMDB_DIR, "data.mdb")
    lock_path = os.path.join(TEMP_LMDB_DIR, "lock.mdb")
    
    print(f"Uploading LMDB files to S3 bucket {BUCKET_NAME} at {LMDB_S3_KEY}...")
    
    # Use multipart upload for large files
    s3.upload_file(
        data_path, 
        BUCKET_NAME, 
        f"{LMDB_S3_KEY}/data.mdb",
        Config=boto3.s3.transfer.TransferConfig(
            multipart_threshold=1024 * 1024 * 8,  # 8MB
            max_concurrency=20,  # Increased concurrency
            multipart_chunksize=1024 * 1024 * 32,  # 32MB per part for faster upload
            use_threads=True
        )
    )
    
    s3.upload_file(lock_path, BUCKET_NAME, f"{LMDB_S3_KEY}/lock.mdb")
    print("Upload complete!")

def print_gpu_info():
    """Print GPU information for debugging."""
    try:
        print("\n==== GPU Information ====")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"Total Memory: {total_mem:.2f} GB")
            
        print("=======================\n")
    except Exception as e:
        print(f"Error getting GPU info: {e}")

if __name__ == "__main__":
    # Print GPU information
    print_gpu_info()
  
    monitor_thread = start_gpu_monitoring()

    
    # Clean cache before starting
    clean_cache()
    
    # Get files to process
    file_keys = list_pdb_files(max_files=None)  # Remove max_files to process all
    
    # Process files with optimized pipeline
    results = process_pdbs_with_pipeline(file_keys)
    
    # Upload results to S3
    upload_lmdb_to_s3()
    print(f"Successfully processed {len(results)} PDB files!")
