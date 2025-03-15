import torch

def clean_embeddings(file1, file2, output_file1, output_file2):
    # Load the data from both files
    data1 = torch.load(file1)
    data2 = torch.load(file2)  # data2 keys are the PDB filenames

    # Extract ids and embeddings from file1
    ids1, embeddings1 = data1["ids"], data1["embeddings"]

    # Extract PDB keys and corresponding embeddings from file2
    raw_ids2 = list(data2.keys()) 
    ids2 = [id_.replace(".pdb", "") for id_ in raw_ids2]  
    embeddings2_dict = {id_.replace(".pdb", ""): data2[id_] for id_ in raw_ids2}  # Updated dict

    # Print first few IDs for debugging
    print("First few IDs from file1:", ids1[:10])  # Print first 10 IDs
    print("First few processed IDs from file2:", ids2[:10])  # Print first 10 stripped PDB keys

    # Find common IDs
    common_ids = set(ids1) & set(ids2)
    num_matching = len(common_ids)  # Count matching embeddings
    print(f"Number of matching embeddings: {num_matching}")


    if num_matching > 0:
        print("First few common IDs:", list(common_ids)[:10])
    else:
        print("No common IDs found. Check if file formats match.")

   
    if num_matching == 0:
        return 0

    # Create index mappings for filtering
    id_to_index1 = {id_: i for i, id_ in enumerate(ids1)}

    # Filter IDs and embeddings
    filtered_ids = sorted(common_ids)  # Sort for consistency
    filtered_embeddings1 = torch.stack([embeddings1[id_to_index1[id_]] for id_ in filtered_ids])
    filtered_embeddings2 = torch.stack([embeddings2_dict[id_] for id_ in filtered_ids])

    # Save cleaned data
    torch.save({"ids": filtered_ids, "embeddings": filtered_embeddings1}, output_file1)
    torch.save({"ids": filtered_ids, "embeddings": filtered_embeddings2}, output_file2)

    print(f"Cleaned data saved to {output_file1} and {output_file2}")

    return num_matching  # Return the count of matching embeddings

# Example usage:
num_matching = clean_embeddings("filtered_output_embeddings.pt", "structure_embeddings_final.pt", 
                                "cleaned_filtered_output_embedding.pt", "cleaned_structure_embeddings_final.pt")
print(f"Returned matching embeddings count: {num_matching}")
