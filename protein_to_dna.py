from Bio.Seq import Seq
import random

# Codon mapping for amino acids
amino_acid_to_codon = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC']
}

def reverse_translate_random(protein_seq):
    dna_seq = ''.join(random.choice(amino_acid_to_codon[aa]) for aa in protein_seq if aa in amino_acid_to_codon)
    return dna_seq

# Read the protein sequences from the file
input_file_path = r"C:\Users\kotsgeo\Documents\GANs\GANs\real_data.fasta"
output_file_path = r"C:\Users\kotsgeo\Documents\GANs\GANs\dna_data.fasta"

# Open the input file
with open(input_file_path, 'r') as input_file:
    with open(output_file_path, 'w') as output_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.strip()
            # Skip header lines or empty ones
            if not line.startswith(">"):
                protein_seq = line
                dna_seq = reverse_translate_random(protein_seq)
                # Write the DNA sequence to the output file
                output_file.write(dna_seq + '\n')

print(f'DNA sequences have been saved to {output_file_path}')