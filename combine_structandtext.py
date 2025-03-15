import os
import gc
import io
import lmdb
import torch
import logging
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
import os
import gc
import torch
import logging
import shutil
from tqdm import tqdm
import concurrent.futures
from functools import partial
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("lmdb_processing.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants and paths
TEMP_DIR = os.path.expanduser("~/gcp-plm/temp_batches")
os.makedirs(TEMP_DIR, exist_ok=True)

# Paths
OUTPUT_DATASET_PATH = "pdb_text_embeddings_with_graphs_merged.pt"
TEMP_OUTPUT_PATH = os.path.join(TEMP_DIR, "temp_output.pt")
lmdb_path = os.path.expanduser("~/lmdb_checkpoints")
text_embeddings_path = "projected_text.pt"

def force_gc():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def process_batch(batch_args):
    batch_id, batch_keys, lmdb_path, key_to_filtered_idx, text_embeddings_path, needed_indices, batch_dir = batch_args
    
    # Initialize batch data dictionary at the beginning
    current_batch_data = {}
    processed_count = 0
    
    try:
        logger.info(f"Starting processing of batch {batch_id}")
        
        needed_embedding_indices = set()
        for _, base_key in batch_keys:
            if base_key in key_to_filtered_idx:
                filtered_idx = key_to_filtered_idx[base_key]
                needed_embedding_indices.add(filtered_idx)
        
        if not needed_embedding_indices:
            logger.warning(f"No valid embedding indices found for batch {batch_id}")
            batch_file = os.path.join(batch_dir, f"batch_{batch_id}.pt")
            torch.save(current_batch_data, batch_file)
            return batch_id, 0
        
        # only the text embeddings we need for this batch
        needed_embedding_indices = sorted(list(needed_embedding_indices))
        idx_to_emb_pos = {needed_embedding_indices[i]: i for i in range(len(needed_embedding_indices))}
        
        # all text embeddings
        try:
            text_data = torch.load(text_embeddings_path, map_location="cpu", weights_only=False)
            batch_embeddings = text_data["projected_text"][needed_indices][needed_embedding_indices].clone()
            del text_data
            gc.collect()
        except Exception as e:
            logger.error(f"Error loading text embeddings for batch {batch_id}: {str(e)}")
            batch_file = os.path.join(batch_dir, f"batch_{batch_id}.pt")
            torch.save(current_batch_data, batch_file)
            return batch_id, 0
        
        # lmdb
        try:
            lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
            
            # Process this batch of keys
            with lmdb_env.begin() as txn:
                for db_key, base_key in batch_keys:
                    try:
                        # Get data from LMDB
                        value = txn.get(db_key)
                        if value is None:
                            continue
                            
                        # Get corresponding text embedding index
                        filtered_idx = key_to_filtered_idx.get(base_key)
                        
                        if filtered_idx is not None and filtered_idx in idx_to_emb_pos:
                            # Process graph data
                            buffer = io.BytesIO(value)
                            with np.load(buffer, allow_pickle=True) as loaded_npz:
                                node_embeddings = loaded_npz['node_embeddings'].astype(np.float32)
                                node_features = loaded_npz['node_features'].astype(np.float32)
                                edge_index = loaded_npz['edge_index']
                                metadata = loaded_npz['metadata'].item()
                                
                                # Get text embedding position in our batch
                                emb_pos = idx_to_emb_pos[filtered_idx]
                                text_emb = batch_embeddings[emb_pos].clone().cpu().numpy().astype(np.float32)
                                
                                # Add to current batch data
                                current_batch_data[base_key] = {
                                    "text_embedding": text_emb,
                                    "graph_data": {
                                        "node_embeddings": node_embeddings,
                                        "node_features": node_features,
                                        "edge_index": edge_index,
                                        "metadata": metadata
                                    }
                                }
                                processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing key {base_key} in batch {batch_id}: {str(e)}")
                        continue
            
            lmdb_env.close()
            
        except Exception as e:
            logger.error(f"Error opening LMDB for batch {batch_id}: {str(e)}")
            if 'lmdb_env' in locals() and lmdb_env is not None:
                lmdb_env.close()
        
        batch_id, batch_keys, lmdb_path, key_to_filtered_idx, text_embeddings_path, needed_indices, batch_dir = batch_args
        
        # After saving batch file:
        batch_file = os.path.join(batch_dir, f"batch_{batch_id}.pt")
        torch.save(current_batch_data, batch_file)
        
        # Clean up memory aggressively. keep runnin out of ram 
        del current_batch_data
        force_gc()
        
        return batch_id, len(current_batch_data)

    
    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {str(e)}")
        #  in case of error, try to save whatever we've processed
        try:
            batch_file = os.path.join(batch_dir, f"batch_{batch_id}.pt")
            torch.save(current_batch_data, batch_file)
        except:
            pass
        return batch_id, len(current_batch_data)

def process_large_dataset_parallel(lmdb_path, text_embeddings_path, output_path, batch_size=500, num_workers=None):
    """Process the dataset in parallel using multiple processes"""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU for the OS
    
    logger.info(f"Starting parallel processing with {num_workers} workers...")
    
    logger.info("First pass: Identifying common keys...")
    
    # Get LMDB keys
    lmdb_keys = set()
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with lmdb_env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key_str = key.decode('utf-8')
                base_key = key_str[:-4] if key_str.endswith(".pdb") else key_str
                lmdb_keys.add(base_key)
        logger.info(f"Found {len(lmdb_keys)} keys in LMDB database")
    finally:
        lmdb_env.close()
    
    # Load only text IDs first (not the embeddings)
    logger.info("Loading only text IDs from embeddings file...")
    text_data = torch.load(text_embeddings_path, map_location="cpu", weights_only=False)
    text_ids = [str(id) for id in text_data["ids"]]
    
    # Create ID to index mapping
    id_to_idx = {text_ids[i]: i for i in range(len(text_ids))}
    
    # Find common keys
    common_keys = set(id_to_idx.keys()) & lmdb_keys
    logger.info(f"Found {len(common_keys)} common keys between LMDB and text embeddings")
    
    # Create filtered list of indices we need
    needed_indices = [id_to_idx[key] for key in common_keys]
    needed_keys = list(common_keys)
    
    # Release memory
    del text_ids, lmdb_keys, text_data
    force_gc()
    
    # Create a mapping from key to filtered index
    key_to_filtered_idx = {needed_keys[i]: i for i in range(len(needed_keys))}
    
    logger.info("Second pass: Processing data in parallel batches...")
    
    # Create temp directory for batch outputs
    batch_dir = os.path.join(TEMP_DIR, f"batches_{os.getpid()}")
    os.makedirs(batch_dir, exist_ok=True)
    
    try:
        # Prepare LMDB environment to get all keys we need to process
        keys_to_process = []
        lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        try:
            with lmdb_env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    base_key = key_str[:-4] if key_str.endswith(".pdb") else key_str
                    if base_key in common_keys:
                        keys_to_process.append((key, base_key))
        finally:
            lmdb_env.close()
        
        total_keys = len(keys_to_process)
        total_batches = (total_keys + batch_size - 1) // batch_size
        logger.info(f"Processing {total_keys} LMDB entries in {total_batches} batches of {batch_size}")
        
        # Create batch arguments for parallel processing
        batch_args = []
        for batch_id in range(total_batches):
            batch_start = batch_id * batch_size
            batch_end = min(batch_start + batch_size, total_keys)
            batch_keys = keys_to_process[batch_start:batch_end]
            
            batch_args.append((
                batch_id, 
                batch_keys, 
                lmdb_path, 
                key_to_filtered_idx, 
                text_embeddings_path, 
                needed_indices,
                batch_dir
            ))
        
        #  batches in parallel
        total_processed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, args) for args in batch_args]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_id, count = future.result()
                total_processed += count
                logger.info(f"Batch {batch_id} completed with {count} entries. Total processed: {total_processed}")
        
        logger.info(f"All batches processed. Total entries: {total_processed}")
        
        return batch_dir
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return None
 

def process_chunk(args):
    chunk_id, files, batch_dir, chunk_dir = args
    chunk_result = {}
    
    for batch_file in files:
        batch_path = os.path.join(batch_dir, batch_file)
        try:
            batch_data = torch.load(batch_path, weights_only=False)
            chunk_result.update(batch_data)
            del batch_data
            gc.collect()
        except Exception as e:
            logger.error(f"Error processing batch file {batch_file} in chunk {chunk_id}: {str(e)}")
    
    # Save chunk result
    chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pt")
    torch.save(chunk_result, chunk_file)
    logger.info(f"Saved chunk {chunk_id} with {len(chunk_result)} entries")
    
    del chunk_result
    gc.collect()
    
    return chunk_id, len(files)

def merge_batch_files(batch_files, batch_dir, output_path, max_memory_entries=10000):
    # Check if we need to start from an existing file
    output_exists = os.path.exists(output_path)
    
    # Initialize a set to track processed entries
    processed_keys = set()
    
    # If output exists, scan it to get keys but don't load content
    if output_exists:
        try:
            logger.info(f"Scanning existing output file for keys: {output_path}")
            # Verify file is valid and loadable
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                try:
                    existing_data = torch.load(output_path, weights_only=False)
                    processed_keys = set(existing_data.keys())
                    del existing_data  # Release memory immediately
                    gc.collect()
                    logger.info(f"Found {len(processed_keys)} existing entries to avoid reprocessing")
                except Exception as e:
                    logger.warning(f"Could not load existing output file: {e}")
                    # Critical change: set output_exists to False if we can't load it
                    output_exists = False
            else:
                logger.warning(f"Output file doesn't exist or is empty, starting fresh")
                output_exists = False
        except Exception as e:
            logger.warning(f"Error checking existing output file: {e}")
            output_exists = False
    
    # Start with an empty dictionary for current batch of merges
    current_batch = {}
    total_processed = len(processed_keys)
    new_entries = 0
    
    # Process all batch files sequentially
    for batch_file in tqdm(batch_files, desc="Merging batch files"):
        batch_path = os.path.join(batch_dir, batch_file)
        
        try:
            # Load the batch data
            batch_data = torch.load(batch_path, weights_only=False)
            
            # Only add entries we haven't processed before
            for key, value in batch_data.items():
                if key not in processed_keys:
                    current_batch[key] = value
                    processed_keys.add(key)
                    new_entries += 1
            
            # Free memory
            del batch_data
            gc.collect()
            
            # Delete the batch file after processing it
            try:
                os.remove(batch_path)
                logger.info(f"Deleted processed batch file: {batch_file}")
            except Exception as e:
                logger.warning(f"Could not delete batch file {batch_file}: {e}")
            
            # Save current batch and clear memory when threshold reached
            if new_entries >= max_memory_entries:
                _save_and_merge_batch(current_batch, output_path, output_exists)
                current_batch = {}
                new_entries = 0
                output_exists = True
                
        except Exception as e:
            logger.error(f"Error in file {batch_file}: {e}")

    
    # Save any remaining entries
    if current_batch:
        _save_and_merge_batch(current_batch, output_path, output_exists)
    
    # Get final count
    total_processed = len(processed_keys)
    logger.info(f"Merge complete! Total entries: {total_processed}")
    
    return total_processed

def _save_and_merge_batch(current_batch, output_path, output_exists):
    if len(current_batch) == 0:
        return
        
    logger.info(f"Saving batch with {len(current_batch)} entries")
    
    # Check if file actually exists and is valid
    file_valid = False
    if output_exists and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            # Try to load a small portion to verify file integrity
            with open(output_path, 'rb') as f:
                pickle.load_buffer = io.BytesIO(f.read(1024))  # Just read the header
            file_valid = True
        except Exception as e:
            logger.warning(f"Output file exists but appears invalid: {e}")
            file_valid = False
    
    if not file_valid:
        # Simple save if no valid existing file
        logger.info(f"Creating new output file: {output_path}")
        torch.save(current_batch, output_path)
        return

    temp_path = f"{output_path}.temp"
    
    try:
        # Load existing data
        existing_data = torch.load(output_path, weights_only=False)
        
        # Update with new batch
        existing_data.update(current_batch)
        
        # Save to temp file
        torch.save(existing_data, temp_path)
        
        # Replace original with temp
        os.replace(temp_path, output_path)
        
        # Free memory
        del existing_data
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        try:
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.info(f"Saving current batch as new output file after merge failure")
                torch.save(current_batch, output_path)
        except Exception as inner_e:
            logger.error(f"Failed to save current batch after merge failure: {inner_e}")

def process_chunk_worker(chunk_info):
    chunk_id, file_chunk, batch_dir, chunk_dir = chunk_info
    chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pt")
    
    try:
        logger.info(f"Processing chunk {chunk_id} with {len(file_chunk)} files")
        chunk_data = {}
        
        # Process each batch file in this chunk
        for batch_file in file_chunk:
            try:
                batch_path = os.path.join(batch_dir, batch_file)
                batch_data = torch.load(batch_path, weights_only=False)
                chunk_data.update(batch_data)
                
                del batch_data
                gc.collect()
                
                # Delete the batch file after processing it
                try:
                    os.remove(batch_path)
                    logger.info(f"Deleted processed batch file: {batch_file}")
                except Exception as e:
                    logger.warning(f"Could not delete batch file {batch_file}: {e}")
                
            except Exception as e:
                logger.error(f"Error in file {batch_file} in chunk {chunk_id}: {e}")
        
        # Save chunk result to disk
        logger.info(f"Saving chunk {chunk_id} with {len(chunk_data)} entries")
        torch.save(chunk_data, chunk_file)
        return chunk_id, len(chunk_data)
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}")
        # Try to save whatever we have
        if 'chunk_data' in locals() and chunk_data:
            try:
                torch.save(chunk_data, chunk_file)
                return chunk_id, len(chunk_data)
            except:
                pass
        # Always return a valid tuple even in case of error
        return chunk_id, 0  # Return 0 entries processed
    


def optimized_parallel_merge(batch_dir, output_path, temp_dir=None, num_workers=None, 
                            chunk_size=10, max_memory_entries=10000):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    if temp_dir is None:
        temp_dir = os.path.dirname(output_path)
    
    # Get all batch files
    batch_files = sorted([f for f in os.listdir(batch_dir) 
                         if f.startswith("batch_") and f.endswith(".pt")])
    
    if not batch_files:
        logger.warning("No batch files found to merge!")
        return 0
    
    logger.info(f"Found {len(batch_files)} batch files to merge using {num_workers} workers")
    
    # Create temp chunk directory
    chunk_dir = os.path.join(temp_dir, f"merge_chunks_{os.getpid()}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    try:
        # Use smaller chunks for better load balancing
        file_chunks = np.array_split(batch_files, min(num_workers * 2, len(batch_files)))
        
        # Convert to list of lists (numpy arrays aren't picklable)
        file_chunks = [chunk.tolist() for chunk in file_chunks]
        
        chunk_infos = [(i, chunk, batch_dir, chunk_dir) for i, chunk in enumerate(file_chunks)]
        
        # Process chunks in parallel with retry mechanism
        chunk_files = []
    
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk_worker, info): info[0] 
                    for info in chunk_infos}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    if result is None:
                        chunk_id = futures[future]  # Get chunk_id from the mapping
                        logger.error(f"Chunk {chunk_id} returned None")
                        continue
                        
                    # Normal case where result is a tuple
                    chunk_id, count = result
                    chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pt")
                    
                    if os.path.exists(chunk_file):
                        chunk_files.append(f"chunk_{chunk_id}.pt")
                        logger.info(f"Chunk {chunk_id} completed with {count} entries")
                    else:
                        logger.warning(f"Chunk {chunk_id} failed to save output file")
                except Exception as e:
                    # Get the chunk_id from our tracking dict
                    chunk_id = futures[future]
                    logger.error(f"Error handling result for chunk {chunk_id}: {e}")

            # Now merge all the chunks using our memory-efficient merger
            logger.info(f"All chunks processed. Merging {len(chunk_files)} chunk files...")
            if chunk_files:
                total_entries = merge_batch_files(chunk_files, chunk_dir, output_path, 
                                                max_memory_entries)
                logger.info(f"Final merge complete! Total entries: {total_entries}")
            else:
                logger.error("No chunk files were successfully created!")
            
    except Exception as e:
        logger.error(f"Error in parallel merge: {e}")
        raise
    finally:
        # Clean up temp files
        try:
            if os.path.exists(chunk_dir):
                logger.info(f"Cleaning up temporary chunk directory: {chunk_dir}")
                shutil.rmtree(chunk_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {chunk_dir}: {e}")
    
    return os.path.exists(output_path)

def main():
    logger.info("Starting parallel processing")
    
    # Get number of CPUs to use
    num_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU for the OS
    logger.info(f"Detected {mp.cpu_count()} CPUs, using {num_cpus} workers")
    
    # Create batch directory
    batch_dir = os.path.join(TEMP_DIR, f"batches_{os.getpid()}")
    os.makedirs(batch_dir, exist_ok=True)
    
    try:
        process_large_dataset_parallel(
            lmdb_path, 
            text_embeddings_path, 
            OUTPUT_DATASET_PATH, 
            batch_size=250, 
            num_workers=num_cpus
        )
        
        # Use optimized merge instead of the old one
        logger.info("Using optimized parallel merge for faster processing...")
        optimized_parallel_merge(
            batch_dir,
            OUTPUT_DATASET_PATH,
            temp_dir=TEMP_DIR,
            num_workers=num_cpus,
            chunk_size=5,  
            max_memory_entries=2000  # Adjust based on available RAM
        )
    finally:
        # Clean up temp files
        if os.path.exists(batch_dir):
            logger.info(f"Cleaning up temporary batch directory: {batch_dir}")
            shutil.rmtree(batch_dir)
    
    logger.info("Parallel processing complete!")

if __name__ == "__main__":
    main()

