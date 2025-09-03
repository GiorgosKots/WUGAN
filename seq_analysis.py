import torch
import numpy as np
import os

def sample_and_analyze(generator=None, epoch=0, device="cpu", pre_generated=None):
    """
    Generate or use pre-generated sequences for analysis
    
    Args:
        generator: Generator model (optional if pre_generated provided)
        epoch: Current epoch number
        device: Device to use
        pre_generated: Pre-generated sequences tensor (optional)
    """
    # Number of samples to generate
    num_samples = 320

    # Decode function
    inv_charmap = {0: 'P', 1: 'A', 2: 'T', 3: 'G', 4: 'C'}

    def decode_sequence(seq):
        indices = np.argmax(seq, axis=1)
        return ''.join([inv_charmap[idx] for idx in indices])

    # Use pre-generated sequences or generate new ones
    if pre_generated is not None:
        generated_sequences = pre_generated
    else:
        # Generate sequences
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, 128).to(device)
            generated_sequences = generator(noise)
        generator.train()

    # Convert to list of strings
    decoded_seqs = [decode_sequence(seq.cpu().numpy()) for seq in generated_sequences]

    return decoded_seqs

def analyze_sequences(seqs):
    # Remove padding sequences
    clean_seqs = [seq.replace('P', '') for seq in seqs]

    # Analysis metrics
    analysis = {
        'total_sequences': len(seqs),
        'sequences_without_padding': len(clean_seqs),

        # Start codon analysis
        'starts_with_atg': sum(1 for seq in clean_seqs if seq.startswith('ATG')),

        # Stop codon analysis
        'ends_with_stop_codon': sum(1 for seq in clean_seqs
                                    if seq.endswith('TAA') or
                                       seq.endswith('TAG') or
                                       seq.endswith('TGA')),

        # Multiple of 3 analysis
        'multiple_of_3': sum(1 for seq in clean_seqs if len(seq) % 3 == 0),

        # All three conditions
        'valid_orfs': sum(1 for seq in clean_seqs
                         if seq.startswith('ATG') and
                            (seq.endswith('TAA') or seq.endswith('TAG') or seq.endswith('TGA')) and
                            len(seq) % 3 == 0),

        # GC Content analysis
        'gc_content': [],
        'avg_gc_content': 0,
        'min_gc_content': 0,
        'max_gc_content': 0
    }

    # Calculate GC content for each sequence
    for seq in clean_seqs:
        # Count G and C
        gc_count = seq.count('G') + seq.count('C')
        # Calculate GC content percentage
        gc_content = (gc_count / len(seq)) * 100 if len(seq) > 0 else 0
        analysis['gc_content'].append(gc_content)

    # Calculate GC content statistics
    if analysis['gc_content']:
        analysis['avg_gc_content'] = np.mean(analysis['gc_content'])
        analysis['min_gc_content'] = np.min(analysis['gc_content'])
        analysis['max_gc_content'] = np.max(analysis['gc_content'])

    # Calculate percentages
    analysis['start_codon_percentage'] = (analysis['starts_with_atg'] / len(clean_seqs)) * 100 if clean_seqs else 0
    analysis['stop_codon_percentage'] = (analysis['ends_with_stop_codon'] / len(clean_seqs)) * 100 if clean_seqs else 0
    analysis['multiple_of_3_percentage'] = (analysis['multiple_of_3'] / len(clean_seqs)) * 100 if clean_seqs else 0
    analysis['valid_orfs_percentage'] = (analysis['valid_orfs'] / len(clean_seqs)) * 100 if clean_seqs else 0

    return analysis

def save_analysis(generated_seqs, epoch, results_dir='results'):
    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Perform analysis
    seq_properties = analyze_sequences(generated_seqs)

    # Create filename
    filename = os.path.join(results_dir, f'analysis_epoch_{epoch}.txt')

    # Write to file
    with open(filename, 'w') as f:
        # First, write all generated sequences
        f.write("Generated Sequences:\n")
        f.write("-" * 20 + "\n")
        for i, seq in enumerate(generated_seqs, 1):
            f.write(f"Sequence {i}: {seq}\n")

        # Then write analysis results
        f.write("\n\nSequence Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Sequences: {seq_properties['total_sequences']}\n")
        f.write(f"Sequences without padding: {seq_properties['sequences_without_padding']}\n")

        f.write("\nStart Codon Analysis:\n")
        f.write(f"Starts with ATG: {seq_properties['starts_with_atg']} "
                f"({seq_properties['start_codon_percentage']:.2f}%)\n")

        f.write("\nStop Codon Analysis:\n")
        f.write(f"Ends with stop codon: {seq_properties['ends_with_stop_codon']} "
                f"({seq_properties['stop_codon_percentage']:.2f}%)\n")

        f.write("\nSequence Length Analysis:\n")
        f.write(f"Sequences multiple of 3: {seq_properties['multiple_of_3']} "
                f"({seq_properties['multiple_of_3_percentage']:.2f}%)\n")

        f.write("\nValid ORFs Analysis:\n")
        f.write(f"Sequences with all conditions (start, stop, multiple of 3): {seq_properties['valid_orfs']} "
                f"({seq_properties['valid_orfs_percentage']:.2f}%)\n")

        f.write("\nGC Content Analysis:\n")
        f.write(f"Average GC Content: {seq_properties['avg_gc_content']:.2f}%\n")
        f.write(f"Minimum GC Content: {seq_properties['min_gc_content']:.2f}%\n")
        f.write(f"Maximum GC Content: {seq_properties['max_gc_content']:.2f}%\n")

# Example usage:
# from seq_analysis import sample_and_analyze, analyze_sequences, save_analysis
# generated_seqs = sample_and_analyze(generator, num_samples=5)
# analysis = analyze_sequences(generated_seqs)
# save_analysis(generated_seqs, epoch=0)
