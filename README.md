[![python 3.8.19](https://img.shields.io/badge/python-3.8.19-brightgreen)](https://www.python.org/)
[![pytorch 2.0.0](https://img.shields.io/badge/pytorch-2.0.0-red)](https://pytorch.org/get-started/previous-versions/)
[![transformers 4.39.2](https://img.shields.io/badge/transformers-4.39.2-blue)](https://pypi.org/)
[![wandb 0.16.5](https://img.shields.io/badge/wandb-0.16.5-orange)](https://pypi.org/)
# Unraveling the Mystery: Predicting the Function, Domain, and GO Term of Each Amino Acid in the Protein Sequence                                              
Work is in progress......
        
                
## About Dataset (Interpro database)                      
**Preprocessing Details**                         
                       
Interpro database ([version 97.0](https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/)) contains 40562 entries and multiple protein function databases. In MetaAI project, the part of protein function prediction focuses on protein family, domain, GO terms and EC number, etc.
In interpro, we preprocessed the `Interpro.xml` file to get the associations between each interpro entry and corresponding different databases (output file is `Interpro_out.txt`).
                   
In addition, we preprocessed the `match_complete.xml` file to obtain the associations between each protein and corresponding different databases and interpro entries (output file is `match_complete_out.txt`).     
After that, we processed the `match_complete_out.txt` to obtain subfiles with protein ids and corresponding labels for each database, and matched corresponding sequences with protein ids in [uniref100](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/Uniref100.fasta.gz).       
      
Thus, the result file contains protein id, sequence, and label is obtained. Up to now, we have completed the processing of `PFAM`, `GENE3D` and `GO` databases. All files have been uploaded to `/home/share/huadjyin/home/chenjunhong/META_AI/databases/`.
                        
---
> **Protein Family:**                 
>  `protein_seq_PFAM.tsv` contains interpro protein id, protein sequence length, sequence and PFAM labels (PFXXXX). The header of the file follows this order. Note that a protein may have multiple PFAM labels, so different PFAM labels are separated by `" "`.            
>         
>       
> **Protein domain:**             
>  `protein_seq_GENE3D.tsv` contains interpro protein id, protein sequence length, sequence and GENE3D labels (G3DSA:XXXX). The header of the file follows this order. Note that a protein may have multiple GENE3D labels, so different GENE3D labels are separated by `" "`.      
>       
> **GO terms:**         
>  `protein_seq_GO.tsv` contains interpro protein id, protein sequence length, sequence and GO terms label (GO:XXXX). The header of the file follows this order. Note that GO describes biological knowledge with respect to three aspects: Molecular Function (MF), Cellular Component (CC) and Biological Process (BP). For a protein, its label may contain MF, CC, and BP simultaneously. So different GO terms are separated by `","` and annotated with (molecular_function/cellular_component/biological_process).           
        
Later, noting that a protein may correspond to multiple PFAM (GENE3D) labels, we found that the reason for this is that each label has a fragment of amino acid sequence belonging to the protein. Therefore, we reprocessed the `match_complete.xml` file to obtain the amino acid sequence start and end position information corresponding to all functional labels under each protein (output file is `fragment_match_complete_out.txt`).

After that, we focused on the functional annotation of PFAM and GENE3D, combining the start and end position information of amino acid sequence fragments (`fragment_match_complete_out.txt`) with the `protein_seq_PFAM.tsv` and `protein_seq_GENE3D.tsv` files described above to obtain the `protein_seq_PFAM_fragment.txt` and `protein_seq_GENE3D_fragment.txt` files.

---
>`protein_seq_PFAM_fragment.txt` contains interpro protein id, complete protein sequence, PFAM labels, start position, end position and fragment annotation.The header of the file follows this order. Note that the protein sequence in this file is complete (not a fragment), it is not segmented according to the location interval, you can use `awk` to get the amino acid sequence fragment of the fixed location of the protein.
>```
>awk '{ $2 = substr($2, $4, $5-$4+1); print }' protein_seq_PFAM_fragment.txt
>```
>The same applies to `protein_seq_GENE3D_fragment.txt` file.

The statistics of the `protein_seq_PFAM_fragment.txt` file are 163,353,266 proteins, 272,102,534 protein fragments (number of file lines), and 20,793 PFAM families. And the statistics of the `protein_seq_GENE3D_fragment.txt` file are 143,196,535 proteins, 266,600,993 protein fragments, and 6,595 GENE3D domains.

**Convert NER labels**  

In order to transform the proteins functional annotation into Named Entity Recognition (NER) task, here we assign functional labels corresponding to each protein in `protein_seq_PFAM_fragment.txt` and `protein_seq_GENE3D_fragment.txt` files with a unique id, and then correspond each amino acid position in the amino acid fragment where the label of each protein is located with a label id to obtain a label id applied to the NER task. 

---
>The preprocessed data files with GENE3D label is located in `/home/share/huadjyin/home/yinpeng/zkx/data/interpro/fragment_info/GENE3D_seq_fragment_out/`. There are a total of 267 subfiles in this directory, and the header of each file is: protein id `\t` sequence `\t` label. Note that the label id corresponding to each amino acid location is separated by `","`. The 267 files contain a total of 143,196,535 proteins, and each line is a unique protein id.
>
>The preprocessed data files with PFAM label is located in `/home/share/huadjyin/home/yinpeng/ljl/data/PFAM_output_1`. Each protein is saved as a `.pkl` file named after the protein ID (sequence, tags, labels).

## Declaration      
This is not an official document, please do not distribute it externally.       
            
            
