import numpy as np
from typing import List, Tuple, Dict, Union

# Genetic code dictionary
GENETIC_CODE = {
    'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L',
    'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
    'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*',
    'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
    'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
    'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
    'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
    'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M',
    'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
    'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
    'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
    'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G'
}

# Amino acid properties
HYDROPHOBIC = set('AILMFVWP')
POSITIVE = set('KRH')
NEGATIVE = set('DE')


def translate_dna(dna: str) -> str:
    """Translate DNA sequence to protein."""
    dna = dna.upper()
    if len(dna) % 3 != 0:
        dna = dna[:-(len(dna) % 3)]
    
    protein = ''
    for i in range(0, len(dna), 3):
        codon = dna[i:i+3]
        if codon in GENETIC_CODE:
            aa = GENETIC_CODE[codon]
            if aa == '*':  # Stop codon
                break
            protein += aa
    return protein


def calculate_peptide_properties(protein: str) -> Dict[str, float]:
    """Calculate AMP-relevant properties of a peptide."""
    if len(protein) == 0:
        return None
    
    length = len(protein)
    
    # Net charge
    pos_charge = sum(1 for aa in protein if aa in POSITIVE)
    neg_charge = sum(1 for aa in protein if aa in NEGATIVE)
    net_charge = pos_charge - neg_charge
    
    # Hydrophobicity percentage
    hydro_count = sum(1 for aa in protein if aa in HYDROPHOBIC)
    hydrophobicity = (hydro_count / length) * 100
    
    # K+R content percentage
    kr_count = sum(1 for aa in protein if aa in 'KR')
    kr_content = (kr_count / length) * 100
    
    return {
        'length': length,
        'net_charge': net_charge,
        'hydrophobicity': hydrophobicity,
        'kr_content': kr_content
    }


def score_peptide(properties: Dict[str, float]) -> Dict[str, float]:
    """Score peptide based on AMP criteria."""
    scores = {
        'length': 1.0 if 5 <= properties['length'] <= 60 else 
                  0.5 if properties['length'] <= 70 else 0,
        'charge': 1.0 if 1 <= properties['net_charge'] <= 12 else 
                  0.5 if 0 <= properties['net_charge'] <= 15 else 0,
        'hydrophobicity': 1.0 if 30 <= properties['hydrophobicity'] <= 60 else 
                          0.5 if 20 <= properties['hydrophobicity'] <= 70 else 0,
        'kr_content': 1.0 if 10 <= properties['kr_content'] <= 50 else 
                      0.5 if 5 <= properties['kr_content'] <= 60 else 0
    }
    return scores


def evaluate_amp_batch(dna_sequences: List[str], return_details: bool = False):
    """
    Evaluate a batch of DNA sequences for AMP properties.

    Args:
        dna_sequences: List of DNA sequences (strings)
        return_details: If True, return detailed metrics

    Returns:
        overall_score: Overall batch score (0-100)
        perfect_amp_percentage: Percentage of sequences that passed all criteria
        details: Dict with detailed metrics (if requested)
    """
    valid_sequences = 0
    passed_all_criteria = 0
    criteria_scores = {'length': 0, 'charge': 0, 'hydrophobicity': 0, 'kr_content': 0}

    for dna in dna_sequences:
        # Check if valid DNA
        if not all(n in 'ATCG' for n in dna.upper()):
            continue

        # Translate and analyze
        protein = translate_dna(dna)
        if len(protein) < 5:  # Changed to 5 to match your criteria
            continue

        properties = calculate_peptide_properties(protein)
        if properties is None:
            continue

        valid_sequences += 1
        scores = score_peptide(properties)

        # Update counters
        for criterion, score in scores.items():
            criteria_scores[criterion] += score

        if all(score >= 1.0 for score in scores.values()):
            passed_all_criteria += 1

    # Calculate final metrics
    if valid_sequences == 0:
        return 0.0 if not return_details else (0.0, 0.0, {'error': 'No valid sequences found'})

    # Normalize scores
    for criterion in criteria_scores:
        criteria_scores[criterion] = (criteria_scores[criterion] / valid_sequences) * 100

    # Overall score
    overall_score = np.mean(list(criteria_scores.values()))
    perfect_amp_percentage = (passed_all_criteria / len(dna_sequences)) * 100

    if return_details:
        details = {
            'overall_score': overall_score,
            'perfect_amp_percentage': perfect_amp_percentage,
            'valid_sequences': valid_sequences,
            'total_sequences': len(dna_sequences),
            'criteria_scores': criteria_scores,
            'passed_all_criteria': passed_all_criteria
        }
        return overall_score, perfect_amp_percentage, details

    return perfect_amp_percentage