from Bio import pairwise2
from Bio.Seq import Seq
from tqdm import tqdm

def calculate_protein_similarity(fasta_file):
    """
    Calculate average sequence similarity from a FASTA file of protein sequences.
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        float: Overall similarity percentage
    """
    # Read sequences from FASTA file
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
        if current_seq:  # Add the last sequence
            sequences.append(current_seq)
    
    print(f"Analyzing {len(sequences)} sequences...")
    
    # Calculate pairwise similarities
    total_similarity = 0
    total_combinations = 0
    
    # Use tqdm for progress bar
    for i in tqdm(range(len(sequences))):
        for j in range(i + 1, len(sequences)):
            seq1 = Seq(sequences[i])
            seq2 = Seq(sequences[j])
            
            # Calculate alignment score
            similarity = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0][2]
            max_length = max(len(seq1), len(seq2))
            total_similarity += (similarity / max_length)
            total_combinations += 1
    
    overall_similarity = (total_similarity / total_combinations) * 100
    
    print(f"\nOverall sequence similarity: {overall_similarity:.2f}%")
    return overall_similarity

def read_fasta_sequences(fasta_file):
    """
    Reads sequences from a FASTA file.
    
    Args:
        fasta_file (str): Path to FASTA file.
        
    Returns:
        list: List of sequences.
    """
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
        if current_seq:  # Add the last sequence
            sequences.append(current_seq)
    return sequences

def calculate_similarity_between_dna(original_fasta, generated_fasta):
    """
    Calculate average sequence similarity between two FASTA files of DNA sequences.
    
    Args:
        original_fasta (str): Path to original FASTA file.
        generated_fasta (str): Path to generated FASTA file.
        
    Returns:
        float: Overall similarity percentage between the two sets of sequences.
    """
    original_sequences = read_fasta_sequences(original_fasta)
    generated_sequences = read_fasta_sequences(generated_fasta)
    
    print(f"Comparing {len(original_sequences)} original sequences with {len(generated_sequences)} generated sequences...")

    total_similarity = 0
    total_combinations = len(original_sequences) * len(generated_sequences)

    for orig_seq in tqdm(original_sequences):
        seq1 = Seq(orig_seq)
        for gen_seq in generated_sequences:
            seq2 = Seq(gen_seq)
            
            # Calculate alignment score
            similarity_alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
            similarity_score = similarity_alignment[2]
            max_length = max(len(seq1), len(seq2))
            normalized_similarity = similarity_score / max_length
            total_similarity += normalized_similarity

    overall_similarity_percentage = (total_similarity / total_combinations) * 100
    
    print(f"\nOverall sequence similarity: {overall_similarity_percentage:.2f}%")
    return overall_similarity_percentage

# Usage:
# similarity = calculate_protein_similarity("your_protein_sequences.fasta")

# similarity = calculate_similarity_between_dna("all_amp.fasta", "valid_sequences.fasta")