from Bio import pairwise2
from Bio.Seq import Seq
from joblib import Parallel, delayed
from tqdm import tqdm

def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file."""
    sequences = []
    with open(fasta_file, 'r') as file:
        current_seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq.upper())
                current_seq = ''
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq.upper())
    return sequences

def sequence_identity(seq1, seq2):
    """Compute % identity for global alignment of two sequences."""
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    aln1, aln2, score, start, end = alignments[0]
    matches = sum(a == b for a, b in zip(aln1, aln2))
    return matches / max(len(seq1), len(seq2)) * 100

def process_one_generated(g, originals, threshold):
    """Get best identity of one generated sequence vs all originals."""
    best_identity = max(sequence_identity(g, o) for o in originals)
    return best_identity < threshold

def novelty_percentage(original_file, generated_file, threshold=80, n_jobs=-1):
    """Calculate % of generated sequences with <threshold% identity to any original (parallel)."""
    originals = read_fasta_sequences(original_file)
    generated = read_fasta_sequences(generated_file)

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_generated)(g, originals, threshold) 
        for g in tqdm(generated, desc="Checking sequences")
    )

    # Count how many were novel
    novel_count = sum(results)
    total_generated = len(generated)
    percentage_novel = (novel_count / total_generated) * 100 if total_generated > 0 else 0

    print(f"{percentage_novel:.2f}")
    return percentage_novel

# Example safe usage:
# novelty_percentage("original_sequences.fasta", "generated_sequences.fasta", threshold=80, n_jobs=16)