## Latent rules

Code for the paper [Supervised Knowledge Aggregation for Knowledge Graph Completion](https://web.informatik.uni-mannheim.de/AnyBURL/betz22aggregation.pdf) from ESWC22


#### Using the Code
The code is build upon the [libKGE](https://github.com/uma-pi1/kge/) library and the rule learner [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/).
The codebase in *sparse/kge* is for the sparse aggregator and *dense/kge* is for the dense aggregator and have to
be installed separately. The most relevant files for the aggregation techniques and training in either 
of the codebases are  *kge/kge/neural\_aggregator.py* for the model implementation and *kge/kge/job/train\_neural\_agg.py*
for training. Configs are located in *./experiments*.

The recommended scenario is one where the number of triples+relations of the KG is large but there is only one or few target relations. The rule learner can efficiently learn regularities of the graph context and the aggregator can effectively combine the learned regularities.



#### Running Sparse on Hetionet
We recommend using a fresh Conda environment for installing all dependencies. The following examples are tested
under Ubuntu os.


```sh
conda create -n sparseenv python=3.7
conda activate sparseenv
```
Then navigate into *sparse/kge* and run:
```sh
pip install -e .
```
For testing if the installation was successful, running the command *kge* from the terminal should output:
```sh
usage: kge [-h] {start,create,resume,eval,valid,test,dump,package} ...
kge: error: the following arguments are required: command
```
Ensure that you are within the folder *sparse/kge* and that the environment is active. Run the command
```sh
kge start examples/hetionet-sparse-agg.yaml --job.device cpu
```
You can use a GPU by using the param "--job.device cuda:0". Now the experiments runs on the training data while using the
validation set for evaluation. A folder will be created in *kge/local/experiments* the folder contains all respective
experimental details. For calculating the final result on the test data, navigate into the folder and run

```sh
kge test . --neural_aggregator.run_test True 
--entity_ranking.metrics_per.head_and_tail True
```
The respective metric for Hetionet is "mean-reciprocal-rank-filtered-tail".

#### Running Dense on Hetionet
Please note to use a new empty environment with a name that is different from "sparseenv" as above.
Installing the codebase then is equivalent to the steps above. Install the code from *code/dense/kge*, and from the 
same folder run 
```sh
kge start examples/hetionet-dense-agg.yaml 
```
where we suggest to use a GPU, i.e, use the GPU flag as described above.

#### Remaining datasets
Running our models for Codex-M, FB15k-237 and WNRR works equivalent to running the code for the Hetionet KG.
However you have to download the respective dataset folder [here](https://www.dropbox.com/home/ESWC22/supplementary-material) first.
Put the respective dataset folder into *code/sparse/kge/data* or *code/dense/kge/data*. After the respective codebase is installed,
you can run our final configuration files of the experiments which you can find in the *experiments* folder. When you 
have trained a model you can directly evaluate it as described above. You can reach a small improvement as in the paper
when picking the checkpoints relation and epoch specific. For doing so continue reading at step 3) below.

#### Reconstructing the full pipeline / Use your own data
We have added the additional code needed for the full pipeline with pre-processing in the folder *utils*.

For the full pipeline, **1)** learn a rule set on a dataset with AnyBURL. Subsequently, **2)** run the script
*code/utils/process.sh* which is preset with  the default parameters for the sparse aggregator. Note that you have 
to modify the paths to your local file systems.  This step will create the output folders *explanations* and
*explanations-processed*. In *explanations* you can find the human-readable input for the aggregators which takes
the input from *explanations-processed*. That is, for the possible queries from a triple on the respective dataset
split, the particular rules that generated the respective candidates are collected. With these inputs the models can
be trained according to the descriptions above. 

**3)** The file *convert.py* can be used to read in a model checkpoint and generates the final rankings for a split of the dataset. These rankings can then be evaluated with
AnyBURL. Please note that the script must be used within the environment that installed one of the codes such that
loading the model checkpoint will work properly. For instance, when you want to generate rankings for the dense
aggregator then please ensure that this codebase is installed.

**4)** Within the folder *utils/combine* a script is
contained that takes as input a directory with multiple checkpoints for valid and test (note that names must be identical except for the string "valid" and "tests")
and then tracks relation- and direction-wise which valid rankings lead to the highest MRR on the valid splits.
This is then used to create a final test ranking.  
