import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
import os

OUTPUT_DIR = "/files/private/notebooks/GANs/WUGAN/campr4"  # Using forward slashes is safer
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
def analyze_sequences(sequences):
    """Analyze GC content of sequences"""
    if not sequences:
        return {
            'avg_gc_content': 0,
            'min_gc_content': 0,
            'max_gc_content': 0
        }
    
    gc_contents = []
    for seq in sequences:
        gc_content = ((seq.count('G') + seq.count('C')) / len(seq)) * 100
        gc_contents.append(gc_content)
    
    return {
        'avg_gc_content': np.mean(gc_contents),
        'min_gc_content': min(gc_contents),
        'max_gc_content': max(gc_contents)
    }

def generate_and_filter_sequences(generator, batch_size=None, num_samples=100000, device="cuda", count_atg=True, add_atg=False):
    """Generate sequences in batches"""
    # Use the batch size that the generator was initialized with
    if batch_size is None:
        batch_size = generator.batch_size if hasattr(generator, 'batch_size') else 64
    
    # Decode function
    inv_charmap = {0: 'P', 1: 'A', 2: 'T', 3: 'G', 4: 'C'}
    def decode_sequence(seq):
        indices = np.argmax(seq, axis=1)
        return ''.join([inv_charmap[idx] for idx in indices])

    valid_sequences = set()
    generator.eval()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            if current_batch_size != batch_size:
                # If we need a different batch size, create a new generator
                temp_generator = type(generator)(
                    generator.n_chars, 
                    generator.seq_len, 
                    current_batch_size, 
                    generator.hidden
                ).to(device)
                temp_generator.load_state_dict(generator.state_dict())
                temp_generator.eval()
                current_generator = temp_generator
            else:
                current_generator = generator

            noise = torch.randn(current_batch_size, 128).to(device)
            generated_sequences = current_generator(noise)

            # Decode sequences
            decoded_seqs = [decode_sequence(seq.cpu().numpy()) for seq in generated_sequences]

            # Clean and filter sequences
            for seq in decoded_seqs:
                # Remove padding
                clean_seq = seq.replace('P', '')

                # Add ATG if specified and not present
                if add_atg and not clean_seq.startswith('ATG'):
                    clean_seq = 'ATG' + clean_seq
                elif not add_atg and count_atg and not clean_seq.startswith('ATG'):
                    continue  # Skip sequences that don't start with ATG when add_atg is False and count_atg is True


                # Check stop codon, length conditions, and uniqueness
                if ((clean_seq.endswith('TAA') or
                     clean_seq.endswith('TAG') or
                     clean_seq.endswith('TGA')) and
                    len(clean_seq) % 3 == 0 and
                    len(clean_seq) > 14):  # Ensure length is more than 17 bases = more that 5 amino acids
                    if clean_seq not in valid_sequences:  # Ensure uniqueness
                        valid_sequences.add(clean_seq)

            # Print progress
            if (i + batch_size) % 10000 == 0:
                print(f"Processed {i + batch_size}/{num_samples} sequences. "
                      f"Found {len(valid_sequences)} valid sequences so far.")

    generator.train()

    # Convert the set to a list for further processing
    valid_sequences = list(valid_sequences)

    # Save analysis of all valid sequences
    analysis = analyze_sequences(valid_sequences)

    # Modify output filename based on ATG setting
    fasta_file = os.path.join(OUTPUT_DIR, "valid_sequences_with_atg.fasta" if add_atg else "valid_sequences.fasta")
    analysis_file = os.path.join(OUTPUT_DIR, "sequence_analysis_with_atg.txt" if add_atg else "sequence_analysis.txt")

    # Save sequences in FASTA format with additional info
    with open(fasta_file, 'w') as f:
        for i, seq in enumerate(valid_sequences, 1):
            # Calculate sequence properties
            gc_content = ((seq.count('G') + seq.count('C')) / len(seq)) * 100
            stop_codon = seq[-3:]

            # Write sequence with detailed header
            f.write(f">sequence_{i} length={len(seq)} GC={gc_content:.2f}% stop_codon={stop_codon}\n{seq}\n")

    # Save detailed analysis results
    with open(analysis_file, 'w') as f:
        f.write(f"Analysis of {len(valid_sequences)} valid sequences:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total sequences generated: {num_samples}\n")
        f.write(f"Valid sequences found: {len(valid_sequences)} ({len(valid_sequences)/num_samples*100:.2f}%)\n")
        f.write("\nSequence Properties:\n")
        if add_atg:
            f.write(f"ATG was added when missing\n")
        else:
            f.write(f"Only natural ATG starts accepted\n")
        f.write(f"Average GC Content: {analysis['avg_gc_content']:.2f}%\n")
        f.write(f"Min GC Content: {analysis['min_gc_content']:.2f}%\n")
        f.write(f"Max GC Content: {analysis['max_gc_content']:.2f}%\n")

        # Add stop codon distribution
        stop_codons = {'TAA': 0, 'TAG': 0, 'TGA': 0}
        for seq in valid_sequences:
            stop_codons[seq[-3:]] += 1

        f.write("\nStop Codon Distribution:\n")
        for codon, count in stop_codons.items():
            percentage = (count / len(valid_sequences)) * 100
            f.write(f"{codon}: {count} ({percentage:.2f}%)\n")

        # Add length distribution
        lengths = [len(seq) for seq in valid_sequences]
        f.write(f"\nLength Statistics:\n")
        f.write(f"Average Length: {np.mean(lengths):.2f}\n")
        f.write(f"Min Length: {min(lengths)}\n")
        f.write(f"Max Length: {max(lengths)}\n")

    print(f"\nGeneration complete!")
    print(f"Found {len(valid_sequences)} valid sequences out of {num_samples} generated")
    print(f"Sequences saved in: {fasta_file}")
    print(f"Analysis saved in: {analysis_file}")

    return valid_sequences, analysis, fasta_file

def convert_dna_fasta_to_protein(input_fasta=None, add_atg=False,  min_amino_acids=3):
    """Convert DNA sequences from FASTA to protein sequences"""
    
    # Set default input file based on add_atg parameter if not provided
    if input_fasta is None:
        input_fasta = os.path.join(OUTPUT_DIR, 
            f"valid_sequences_{'with_atg' if add_atg else 'natural'}.fasta")
    
    # Set output filename based on input
    base_name = os.path.basename(input_fasta)
    output_fasta = os.path.join(OUTPUT_DIR, base_name.replace('.fasta', '_proteins.fasta'))
    
    # Set output filename based on input
    output_fasta = input_fasta.replace('.fasta', '_proteins.fasta')
    
    protein_count = 0
    
    # Open output file for proteins
    with open(output_fasta, 'w') as protein_file:
        # Read DNA sequences from input FASTA
        for record in SeqIO.parse(input_fasta, "fasta"):
            # Get DNA sequence
            dna_seq = str(record.seq)
            
            # Translate to protein
            try:
                protein_seq = str(Seq(dna_seq).translate(to_stop=True))
                
                # Write protein sequence if it's valid and long enough
                if protein_seq and len(protein_seq) >= min_amino_acids:
                    protein_count += 1
                    
                    # Get original DNA sequence info from header
                    original_info = record.description.split()
                    dna_length = next(info.split('=')[1] for info in original_info if 'length=' in info)
                    gc_content = next(info.split('=')[1] for info in original_info if 'GC=' in info)
                    
                    # Write protein sequence with header
                    protein_file.write(f">protein_{protein_count} length={len(protein_seq)} "
                                     f"original_dna_length={dna_length} "
                                     f"original_gc={gc_content} "
                                     f"original_id={record.id}\n{protein_seq}\n")
            except Exception as e:
                print(f"Error processing sequence {record.id}: {str(e)}")
                continue
    
    print(f"Converted {protein_count} DNA sequences to proteins")
    print(f"Protein sequences saved in: {output_fasta}")
    
    return protein_count, output_fasta

### 
# from sequence_generator import generate_and_filter_sequences, convert_dna_fasta_to_protein


# # Example usage:
# if __name__ == "__main__":
#     # Generate DNA sequences
#     valid_sequences, analysis, fasta_file = generate_and_filter_sequences(
#         generator=your_generator,
#         num_samples=100000,
#         add_atg=True  # or False
#     )
    
#     # Convert to proteins
#     num_proteins, protein_file = convert_dna_fasta_to_protein(
#         input_fasta=fasta_file,  # Uses the fasta file from generation
#         add_atg=True  # should match the generation parameter
#     )

# or #
# # Generate sequences and convert to proteins in one go
# sequences, analysis, dna_file = generate_and_filter_sequences(generator, add_atg=True)
# num_proteins, protein_file = convert_dna_fasta_to_protein(input_fasta=dna_file, add_atg=True)
