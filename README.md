 # <b>  DNA Sequencing Using Machine Learning Algorithms
  
<br></br>
# <a id = 'ProblemStatement'>Problem Statement</b></a>
High-throughput Next-Generation Sequencing (NGS) has significantly advanced our understanding of biology. However, ensuring the accuracy of recorded data remains a challenging yet essential step, as the interpretation of the human genome heavily depends on the precision of sequencing results. Leveraging traditional machine learning techniques alongside deep learning neural networks offers a robust approach to rigorously evaluate sequencing outcomes and enhance the accuracy of genomic and transcriptomic data.

This project uses machine learning to improve the classification of human family genes based on DNA sequences. A dataset comprising 4,380 human DNA sequences and 7 family genes was utilized to train and evaluate various models using multiclass classification techniques. Traditional machine learning algorithms, including K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machines (SVM), were employed to establish baseline accuracy. Subsequently, deep neural networks with optimized architectures were developed to enhance classification performance further.

The primary objective of this project is to accurately classify human family genes based on a given set of DNA sequences, showcasing the potential of combining traditional and deep learning approaches in analyzing and interpreting genomic data.

---
<br></br>
# <a id = 'Content'> Content </b></a>

- [Problem Statement](#ProblemStatement)
- [Content](#Content)    
- [Repo Structure](#RepoStructure)    

    - [Data Dictionary](#ddict)
    - [Background](#Background)
    - [Statistical Models: Methodology and Concepts](#ModelingMethodology)    
    	- [1. Random Forest](#RandomForest)	
    	- [2. Neural Networks](#NeuralNetworks)	
    - [Results](#Results)    
<!--     - [Conclusion](#Conclusion)
    - [Recommendations](#Recommendations)
    - [References](#references) -->



---
# <a id = 'RepoStructure'> Repo Structure </b></a>
## notebooks/ <br />

*Setp 1: Exploratory Data Analysis:*\
&nbsp; &nbsp; &nbsp; __ [Exploratory Data_Analysis_EDA.ipynb](notebooks/1ExploratoryDataAnalysisEDA.ipynb)<br />

*Setp 2: DNA Sequence Dataset Processing:*\
&nbsp; &nbsp; &nbsp; __ [Generate kmer dataset.ipynb](notebooks/2GeneratekmerDataset.ipynb)<br />

*Setp 3: Traditional Machine Learning Models: Classifiers*\
&nbsp; &nbsp; &nbsp; __ [KNeighborsClassifier.ipynb](notebooks/3KNeighborsClassifier.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [RandomForest.ipynb](notebooks/4RandomForest.ipynb)<br />


*Setp 4: Feed-Forward Neural Networks: Classifiers*\
&nbsp; &nbsp; &nbsp; __ [FeedForward Neural Networks.ipynb](notebooks/5FeedForwardNeuralNetworks.ipynb)<br />
/>



## datasets/<br />
*Unprocessed data collected from sub Reddits:*\
&nbsp; &nbsp; &nbsp; __ [human.txt](datasets/human.txt)<br />


---
---
# <a id = 'ddict'>Dataset <b>Dictionary</b></a>


|feature name|data type| possible values | represents| description | reference|
|---|---|---|---|---|---|
| Sequence |*object*| A, T, G, C | DNA sequence|    |   |
| Class|*integer*|0 |  G protein-coupled receptors (GPCRs)| G-protein-coupled receptors (GPCRs) are the largest and most diverse group of membrane receptors in eukaryotes. These cell surface receptors act like an inbox for messages in the form of light energy, peptides, lipids, sugars, and proteins| [[link]](https://www.nature.com/scitable/topicpage/gpcr-14047471/) |
|  |*integer*|1 |  Tyrosine kinase| a large multigene family with particular relevance to many human diseases, including cancer|[[link]](https://www.nature.com/articles/1203957) |
|  |*integer*|2 |  Protein tyrosine phosphatases| Protein tyrosine phosphatases are a group of enzymes that remove phosphate groups from phosphorylated tyrosine residues on proteins| [[link]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiI9omSsfP1AhVeJ0QIHbQbAF8QFnoECAcQAw&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FProtein_tyrosine_phosphatase&usg=AOvVaw26Gc_GqosG5hJnZu1uf4cy)|
|  |*integer*|3 |  Protein tyrosine phosphatases (PTPs)| to control signalling pathways that underlie a broad spectrum of fundamental physiological processes | [[link]](https://pubmed.ncbi.nlm.nih.gov/17057753/)|
|  |*integer*|4 |  Aminoacyl-tRNA synthetases (AARSs)| responsible for attaching amino acid residues to their cognate tRNA molecules, which is the first step in the protein synthesis | [[link]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC29805/)|
|  |*integer*|5 |  Ion channels| Ion channels are the pathways for the passive transport of various inorganic ions across a membrane| [[ref]](https://www.frontiersin.org/articles/10.3389/fgene.2019.00399/full) |
|  |*integer*|6 |  Transcription Factor| Transcription factors are proteins involved in the process of converting, or transcribing, DNA into RNA | [[link]](https://www.nature.com/scitable/definition/transcription-factor-167/)|

---
---
# <a id = 'Background'>Background</a> 
## 1. <a id = 'api'> DNA sequencing </a> 
DNA (Deoxyribonucleic Acid) sequencing is identifying the precise order of nucleotides within a DNA molecule. This involves determining the sequence of the four bases: Adenine (A), Guanine (G), Cytosine (C), and Thymine (T). DNA sequencing is a vital technique in biology, serving as a fundamental step in understanding the genetic basis of various diseases. The figure below highlights the core concept of DNA sequencing along with its key applications. [[ref]](https://www.nist.gov/patents/nucleic-acid-sequencer-electrically-determining-sequence-nitrogenous-bases-single-stranded). 


---
# <a id = 'ModelingMethodology'>Statistical Models: Methodology and Concepts</b></a>


## <a id = 'RandomForest'>1. Random Forest</b></a>
Random Forest is a supervised machine learning algorithm and a key ensemble method. It addresses overfitting issues commonly seen in decision trees by building a large ensemble of bootstrap trees and combining their results. This blog delves into the foundational concepts behind both Random Forest Classifiers and Regressors.

In the bagging technique, all features are used, but the dataset's observations (rows) vary, leading to some correlation among the bootstrap trees and resulting in high variance. Random Forest reduces this correlation by randomly selecting subsets of features, ensuring not all features are included in every decision tree. As a result, Random Forest enhances the Bagging technique, creating a low-correlation ensemble of decision trees for more reliable predictions.

`from sklearn.ensemble import RandomForestClassifier`




## <a id = 'NeuralNetworks'>2. Neural Networks</b></a>

Dendrites, axons, and cell bodies may not be familiar terms to everyone, but understanding the complexity of neural networks in the brain provides a helpful analogy for grasping how computers are trained to solve problems. This comparison offers an intuitive starting point for exploring deep neural networks.

The process can be explained as follows: synaptic strengths, represented by weights (w), are adjustable and determine the influence of incoming signals. Dendrites carry these signals to the cell body, where they are summed. If the total exceeds a certain threshold, the neuron "fires," sending a signal through its axon. In computational models, the exact timing of these signals is not considered critical; instead, the firing frequency conveys information. Using this "rate code" approach, the firing rate is modeled with an activation function (f), allowing the network to process and transmit information effectively.

Inspired by biological neural networks, this analogy serves as a foundational concept for understanding deep learning.





# <a id = 'Results'>Results</b>

In this project, all posts are divided into training and testing sets with a 75% to 25% split, respectively. For each dataset, performance metrics such as accuracy, precision, recall, and F1 score are calculated. Additionally, the values for false positives, false negatives, true positives, and true negatives are reported. Their definitions are provided in the following equations. [[ref](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)]:



     





