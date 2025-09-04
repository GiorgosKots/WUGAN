def read_fasta_sequences(fasta_file):
    """Read sequences from a FASTA file (your implementation)."""
    sequences = []
    with open(fasta_file, 'r') as file:
        current_seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):  # header
                if current_seq:
                    sequences.append(current_seq.upper())  # keep consistent
                current_seq = ''
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq.upper())
    return sequences


def percentage_unique_generated(original_file, generated_file):
    """Print % of generated sequences that are NOT in original dataset."""
    original = set(read_fasta_sequences(original_file))
    generated = read_fasta_sequences(generated_file)

    unique_generated = [seq for seq in generated if seq not in original]
    total_generated = len(generated)
    percentage_unique = (len(unique_generated) / total_generated) * 100 if total_generated > 0 else 0

    # Print only the percentage
    print(f"Percentage of unique generated sequences: {percentage_unique:.2f}")


#example usage
# percentage_unique_generated("original_sequences.fasta", "generated_sequences.fasta")