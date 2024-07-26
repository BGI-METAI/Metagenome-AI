## About Dataset (Interpro database)

Interpro database ([version 97.0](https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/)) contains 40562 entries and multiple protein function databases. In MetaAI project, the part of protein function prediction focuses on protein family and GO terms, etc.

## **File Descriptions**

`protein_seq_PFAM.tsv`: file of PFAM integrated by InterPro

`protein_seq_GENE3D.tsv`:file of GENE3D integrated by InterPro

`protein_seq_GO.tsv`:file of GO terms integrated by InterPro

`uniprotkb_AND_reviewed_true_2024_06_24_filter.tsv`: high-quality file of GO terms integrated by swiss-prot

## **Preprocessing Details**

 In interpro, we preprocessed the `match_complete.xml` file to obtain subfiles with protein ids and corresponding labels for each database (such as PFAM), and matched corresponding sequences with protein ids in [uniref100](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/Uniref100.fasta.gz). Thus, the result file contains protein id, sequence, and label is obtained. Up to now, we have completed the processing of `PFAM`, `GENE3D` and `GO` databases (`protein_seq_PFAM.tsv`, `protein_seq_GENE3D.tsv`, `protein_seq_GO.tsv`). All files have been uploaded to `/home/share/huadjyin/home/chenjunhong/META_AI/databases/`.



Later, noting that a protein may correspond to multiple PFAM (GENE3D) labels, we found that the reason for this is that each label has a
fragment of amino acid sequence belonging to the protein. Therefore, we reprocessed the `match_complete.xml` file to obtain the 
amino acid sequence start and end position information corresponding to all functional labels under each protein. After that, 
we focused on the functional annotation of PFAM and GENE3D, combining the start and end position information of amino 
acid sequence fragments with the `protein_seq_PFAM.tsv` and `protein_seq_GENE3D.tsv` files 
described above to obtain the `protein_seq_PFAM_fragment.txt` and `protein_seq_GENE3D_fragment.txt` files.

---
> **Protein Family:**
> Contains `163,353,266` protein sequences and `20,793` PFAM labels. We conducted statistics on the length of PFAM proteins and found that the maximum length is `45,354` and the minimum length is `8`(the protein length distribution is shown in Figure 1 and  Figure 2). In addition, we performed a statistical analysis of the size of the PFAM category and the result file was located in `/home/share/huadjyin/home/yinpeng/zkx/data/interpro/interpro_result/pFAM_info_out.txt`(the distribution of protein families and protein numbers is shown in Figure 3).
>
> <img src="C:\Users\zhangkexin2\AppData\Roaming\Typora\typora-user-images\image-20240726135721387.png" alt="image-20240726135721387" style="zoom: 67%;" />
>
> ​                                                                               Figure 1. The distribution of PFAM whole protein length (full range)
>
> <img src="C:\Users\zhangkexin2\AppData\Roaming\Typora\typora-user-images\image-20240726141301941.png" alt="image-20240726141301941" style="zoom:67%;" />
>
> ​                                                                        Figure 2. The distribution of PFAM whole protein length (0-1000 range)
>
> <img src="C:\Users\zhangkexin2\AppData\Roaming\Typora\typora-user-images\image-20240726140451452.png" alt="image-20240726140451452" style="zoom:67%;" />
>
> ​                                                                                         Figure 3. The distribution of protein families and protein numbers
>
> `protein_seq_PFAM.tsv` contains interpro protein id, protein sequence length, sequence and PFAM labels (PFXXXX).
> The header of the file follows this order. Note that a protein may have multiple PFAM labels, so different PFAM labels are separated by `" "`.            
>
> **Protein domain:**    
>
> Contains `143,196,535` protein sequences and `6595` GENE3D labels.         
> `protein_seq_GENE3D.tsv` contains interpro protein id, protein sequence length, sequence and GENE3D labels (G3DSA:XXXX). 
> The header of the file follows this order. Note that a protein may have multiple GENE3D labels, so different GENE3D labels are separated by `" "`.      
>
> **GO terms:**     
>
> Contains `120,570,411` protein sequences and `5772` GO terms.    
> `protein_seq_GO.tsv` contains interpro protein id, protein sequence length, sequence and GO terms label (GO:XXXX).
> The header of the file follows this order. Note that GO describes biological knowledge with respect to three aspects: Molecular Function (MF),
> Cellular Component (CC) and Biological Process (BP). For a protein, its label may contain MF, CC, and BP simultaneously.
> So different GO terms are separated by `","` and annotated with (molecular_function/cellular_component/biological_process). However, according to our investigation, although the GO term data integrated by interpro is large in volume, the quality is not high. Therefore, if you want to conduct pre-experiments, it is recommended to use the GO term data from swiss-prot (see Supplementary Content).

## **Independent test set description**s

For PFAM, in order to partition the data set while being comparable to other baseline methods, we collated several comparison methods for predicting PFAM tasks, such as protENN and proteinvec. For protENN, because its data does not use the uniprot protein id as a unique identifier, we converted the id according to the mapping relationship of uniprot and filtered out the data that did not match (see the following table for details). After that, we merged the integrated test set data to obtain 143017 protein data. The integrated file path is `/home/share/huadjyin/home/zhangkexin2/data/benchmark_merge_data/PFAM/protvec_protenn_all_test_protein_merge.txt`. Further, we extend this test data using [uniref50]([* in UniRef search (66075574) | UniProt](https://www.uniprot.org/uniref?query=*&facets=identity%3A0.5)) to form a ratio of 6:2:2 for the training, validation and test data.

| model           | independent test data | filtered |
| --------------- | --------------------- | -------- |
| protENN-cluster | 21293                 | 21022    |
| protENN-random  | 126171                | 123815   |
| proteinvec      | 2295                  | 2295     |
| merged          | -                     | 143017   |

## **Supplementary Content**

* GO terms

We prepared a dataset for GO function prediction, which comes from swiss-prot, with a total of `551,192` sequences and `28,831` classes. The original file path is `/home/share/huadjyin/home/zhangkexin2/data/proteinNER/GO/swiss-prot/preprocess/uniprotkb_AND_reviewed_true_2024_06_24_filter.tsv`.But because some classes are small in size, we filtered them out and kept only those with more than `50` proteins. In addition, only proteins with a length of less than `1000` amino acids were selected for the experiment. The number of proteins after preprocessed was `531,019`, with `6254` classes. After that, we divided the dataset described above into the train set, validation set and test set according 6:6:2, these files path are `/home/share/huadjyin/home/zhangkexin2/data/proteinNER/GO/swiss-prot/filter_class/*_data_path.txt`. Each row represents the path to a protein pkl file.





