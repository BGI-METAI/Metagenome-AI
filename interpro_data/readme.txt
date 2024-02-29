Interpro Database Preprocessd Details
 
Interpro database (version 97.0, URL: https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/) contains 40562 entries and multiple protein function databases. In MetaAI project, the part of protein function prediction focuses on protein family, domain, GO terms and EC number, etc.
In interpro, we preprocessed the Interpro.xml file to get the associations between each interpro entry and corresponding different databases (output file is Interpro_out.txt). In addition, we preprocessed the match_complete.xml file to obtain the associations between each protein and corresponding different databases and interpro entry (output file is match_complete_out.txt). After that, we processed the match_complete_out.txt to obtain subfiles with protein ids and labels of the same database, and matched corresponding sequences with protein ids in uniref100 (URL: https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/Uniref100.fasta.gz). Thus, the result file of protein id, sequence, and label is obtained. Up to now, we have completed the processing of PFAM, GENE3D and GO terms.

About protein family:
protein_seq_PFAM.tsv contains interpro protein id, protein sequence length, sequence and PFAM labels (PFXXXX). The header of the file follows this order. Note that a protein may have multiple PFAM labels, so different GENE3D labels are separated by " ".


About protein domain:
protein_seq_GENE3D.tsv contains interpro protein id, protein sequence length, sequence and GENE3D labels (G3DSA:XXXX). The header of the file follows this order. Note that a protein may have multiple GENE3D labels, so different GENE3D labels are separated by " ".


About protein GO terms:
protein_seq_GO.tsv contains interpro protein id, protein sequence length, sequence and GO terms label (GO:XXXX). The header of the file follows this order. Note that GO describes biological knowledge with respect to three aspects: Molecular Function (MF), Cellular Component (CC) and Biological Process (BP). For a protein, its label may contain MF, CC, and BP simultaneously. So different GO terms are separated by "," and annotated with (molecular_function/cellular_component/biological_process).
