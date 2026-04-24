import io
import pandas as pd
from Bio import SeqIO

def parse_fasta(fasta_content):
    """Parse FASTA content from a string or file-like object."""
    fasta_io = io.StringIO(fasta_content)
    records = list(SeqIO.parse(fasta_io, "fasta"))
    return [(record.id, str(record.seq)) for record in records]

def embeddings_to_csv(labels, embeddings):
    """Convert embeddings to a CSV string.
    
    Args:
        labels (list): List of sequence IDs.
        embeddings (torch.Tensor): Tensor of shape (N, D).
    """
    df = pd.DataFrame(embeddings.cpu().numpy(), index=labels)
    df.index.name = "variant"
    return df.to_csv()
