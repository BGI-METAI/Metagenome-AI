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
For convenience, we store each protein sequence and corresponding label (Protein Family/Protein domain/GO Terms) as a pickle file, and named with `"PROTEIN_ID.pkl""`. For each pickle file contain `seq`, `token_label`, `start` and `end` fields. We use `BIO` strategy to generate labels for NER tasks.                          
> **Protein Family:** Contains `163,353,266` protein sequences and `20,795` PFAM families. We divided the data set into `three` groups according to the length of the protein sequence, namely `less than 250`, `greater than or equal to 250 and less than 500`, and `greater than or equal to 500`.               
> The preprocessed pickle file is stored in `/home/share/huadjyin/home/zhangchao5/dataset/version2/pfam/pkls`, which contains `20,975` folders.            

<table>
<tr>
<th>group name</th>
<th>sequence number</th>
<th>PFAM number</th>
</tr>
<tr>
<td>seq<250</td>
<td>54,153,989</td>
<td>20,414</td>
</tr>
<tr>
<td>seq>=250, seq<500</td>
<td>98,803,842</td>
<td>18,654</td>
</tr>
<tr>
<td>seq>=500</td>
<td>82,649,256</td>
<td>16,795</td>
</tr>
</table>

> **Protein domain:** Contains `***` protein sequences and `***` GENE3D labels. We divided the data set into `***` groups according to the length of the protein sequence, namely `***`, `***`, and `***`.           
>  The preprocessed pickle file is stored in `***`, which contains `***` folders.       
>       
> **GO terms:** Contains `***` protein sequences and `***` Go-TERMs. We divided the data set into `***` groups according to the length of the protein sequence, namely `***`, `***`, and `***`.       
>  "The preprocessed pickle file is stored in `***`, which contains `***` folders.           
        
## Training        
**1. PFAM**    
> Firstly, we selected `20` protein families from the data with protein sequence length `less than 250` to form a subset (`6,726,147` protein sequence) and used it for model training.         
<table>
<tr>
<td>Protein Family Name</td>
<td>PF00072</td>
<td>PF00440</td>
<td>PF00005</td>
<td>PF00115</td>
<td>PF00583</td>
<td>PF04542</td>
<td>PF00486</td>
<td>PF00293</td>
<td>PF08281</td>
<td>PF01381</td>
</tr>
<tr>
<td>Sequence Number</td>
<td>867,738</td>
<td>635,459</td>
<td>580,588</td>
<td>504,575</td>
<td>469,790</td>
<td>344,739</td>
<td>332,982</td>
<td>309,546</td>
<td>304,195</td>
<td>298,001</td>
</tr>
</table>
<table>
<tr>
<td>Protein Family Name</td>
<td>PF00196</td>
<td>PF00903</td>
<td>PF00392</td>
<td>PF00106</td>
<td>PF13561</td>
<td>PF00528</td>
<td>PF07883</td>
<td>PF13302</td>
<td>PF00578</td>
<td>PF00156</td>
</tr>
<tr>
<td>Sequence Number</td>
<td>284,671</td>
<td>274,281</td>
<td>243,545</td>
<td>190,808</td>
<td>190,395</td>
<td>188,950</td>
<td>183,997</td>
<td>178,398</td>
<td>172,073</td>
<td>171,416</td>
</tr>
</table>
         
> Then, we randomly divide the subset into training set and test set according to `8:2` (random seed: `42`).                 
        
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
            
            
