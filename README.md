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
      
Thus, the result file contains protein id, sequence, and label is obtained. Up to now, we have completed the processing of `PFAM`, `GENE3D` and `GO` databases.
                        
---            
In order to transform the proteins functional annotation into Named Entity Recognition (NER) task, here we assign functional labels with a unique id, and then correspond each amino acid position in the amino acid fragment where the label of each protein is located with a label id to obtain a label id applied to the NER task.                    
For convenience, we store each protein sequence and corresponding label (Protein Family/Protein domain/GO Terms) as a pickle file, and named with `"PROTEIN_ID.pkl""`. For each pickle file contain `seq` and `label` fields.                          
> **Protein Family:** Contains `163,353,266` protein sequences and `20,793` PFAM families. We removed the protein sequence length more than 800.               
> The preprocessed pickle file is stored in `/home/share/huadjyin/home/zhangchao5/dataset/pfam`, which contains `pfam.train`, `pfam.test` and `pfam.valid` folders.            
>       
> **Protein domain:** Contains `143,196,535` protein sequences and `6,595` GENE3D labels. We removed the protein sequence length more than 1000.           
>  The preprocessed pickle file is stored in `/home/share/huadjyin/home/zhangchao5/dataset/gene3d`, which contains `gene3d.train`, `gene3d.test` and `gene3d.valid` folders.       
>       
> **GO terms:** Contains `***` protein sequences and `***` Go-TERMs. We removed the protein sequence length more than 1000.       
>  "The preprocessed pickle file is stored in `/home/share/huadjyin/home/zhangchao5/dataset/go_term`, which contains `go_term.train`, `go_term.test` and `go_term.valid` folders.           
        
## Metrics          
**1. Token Level**          
$$Accuracy=\frac{CorrectlyClassifiedTokens}{TotalTokens}$$                          
              
**2. Entity Level**      
$$Precision=\frac{CorrectlyClassifiedEntities}{PredictedEntities}$$           
           
**3. Recall (Entity Level)**       
$$Recall=\frac{CorrectlyClassifiedEntities}{ActuallyEntities}$$         
           
**4. Mean Intersection over Union (mIoU)**                       
The mIoU calculation formula for each predicted entity is as follows,           
$$*******$$
           
## Declaration      
This is not an official document, please do not distribute it externally.       
            
            
