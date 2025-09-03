from joblib import Parallel, delayed
from Bio.Seq import Seq
from Bio import Align
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file"""
    sequences = []
    with open(fasta_file, 'r') as file:
        current_seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ''
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences

def similarity_score(seq1, seq2):
    """Calculate similarity between two sequences using the new PairwiseAligner"""
    # Create aligner with settings equivalent to globalxx
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0
    
    # Perform alignment
    alignments = aligner.align(seq1, seq2)
    
    # Get the best alignment score
    best_score = alignments[0].score
    
    # Calculate normalized similarity (0-1)
    max_possible_score = max(len(seq1), len(seq2))
    similarity = best_score / max_possible_score
    
    return similarity

def calculate_protein_similarity_parallel(fasta_file, n_jobs=-1):
    """
    Parallelized similarity calculation using all CPU cores.
    n_jobs=-1 means use all available cores.
    """
    sequences = read_fasta_sequences(fasta_file)
    n_sequences = len(sequences)
    
    if n_sequences == 0:
        print("No sequences found in the file.")
        return 0
    
    if n_sequences == 1:
        print("Only one sequence found. Cannot calculate similarity.")
        return 100.0
    
    print(f"Analyzing {n_sequences} sequences on {n_jobs if n_jobs > 0 else 'all'} cores...")
    
    # Calculate total number of comparisons
    total_comparisons = (n_sequences * (n_sequences - 1)) // 2
    print(f"Total comparisons to perform: {total_comparisons}")
    
    # Prepare all pairs (upper triangular matrix)
    pairs = [(Seq(sequences[i]), Seq(sequences[j]))
             for i in range(n_sequences)
             for j in range(i + 1, n_sequences)]
    
    # Perform parallel computation with progress bar
    with tqdm_joblib(tqdm(desc="Calculating similarities", total=len(pairs))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(similarity_score)(s1, s2) for s1, s2 in pairs
        )
    
    # Calculate statistics
    if results:
        overall_similarity = (sum(results) / len(results)) * 100
        min_similarity = min(results) * 100
        max_similarity = max(results) * 100
        
        print(f"\n--- Similarity Statistics ---")
        print(f"Overall sequence similarity: {overall_similarity:.2f}%")
        print(f"Minimum similarity: {min_similarity:.2f}%")
        print(f"Maximum similarity: {max_similarity:.2f}%")
        print(f"Number of sequences: {n_sequences}")
        print(f"Number of comparisons: {len(results)}")
    else:
        overall_similarity = 0
        print("No comparisons were made.")
    
    return overall_similarity

# Usage:
# similarity = calculate_protein_similarity_parallel("your_sequences.fasta")
