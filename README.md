# Framework for Protein Function/Domain/GO-terms Prediction                     
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
>  `protein_seq_PFAM.tsv` contains interpro protein id, protein sequence length, sequence and PFAM labels (PFXXXX). The header of the file follows this order. Note that a protein may have multiple PFAM labels, so different GENE3D labels are separated by `" "`.            
>         
>       
> **Protein domain:**             
>  `protein_seq_GENE3D.tsv` contains interpro protein id, protein sequence length, sequence and GENE3D labels (G3DSA:XXXX). The header of the file follows this order. Note that a protein may have multiple GENE3D labels, so different GENE3D labels are separated by `" "`.      
>       
> **GO terms:**         
>  `protein_seq_GO.tsv` contains interpro protein id, protein sequence length, sequence and GO terms label (GO:XXXX). The header of the file follows this order. Note that GO describes biological knowledge with respect to three aspects: Molecular Function (MF), Cellular Component (CC) and Biological Process (BP). For a protein, its label may contain MF, CC, and BP simultaneously. So different GO terms are separated by `","` and annotated with (molecular_function/cellular_component/biological_process).           
        
            
## Declaration      
This is not an official document, please do not distribute it externally.       
            
            