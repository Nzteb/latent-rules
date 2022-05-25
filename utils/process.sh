#!/bin/bash


data_dir="/path/to/dataset/folder"
rule_file="/path/to/file/with/rules"
anyburl_path="/path/to/AnyBURL/AnyBURL-PT.jar"
# this is the foler where everything is stored, it will e in the dataset folder
output_folder_name="output-folder-name/"
# preprocess_explanations.py location
process_explanations_path="/path/to/preprocess_explanations.py"
worker_threads=10

output_dir=$data_dir$output_folder_name


echo "$output_dir"

if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
else
  echo "Output folder exists already" 1>&2
  echo "$output_dir"
  #exit 1
fi

if [ ! -f "$rule_file" ]; then
  echo "Found no rules in" 1>&2
  echo "$rule_file"
  exit 1
fi


apply_config="config-apply-explanations.properties"

explanations_folder="${output_dir}explanations/"
processed_explanations_folder="${output_dir}explanations-processed/"


if [ ! -d "$explanations_folder" ]; then
    mkdir "$explanations_folder"
fi


#write explanations for train
PATH_TRAINING="${data_dir}train.txt"
PATH_VALID="${data_dir}empty.txt"
PATH_TEST="${data_dir}train_target.txt"
PATH_OUTPUT="${explanations_folder}train-explanations"
apply_config="${explanations_folder}config-apply-explanations-train.properties"
PATH_RULE_INDEX="${explanations_folder}rule-index-train"

/bin/cat <<EOM >$apply_config
PATH_TRAINING  = $PATH_TRAINING
PATH_VALID     = $PATH_VALID
PATH_TEST      = $PATH_TEST

PATH_RULES      = $rule_file
PATH_OUTPUT     = $PATH_OUTPUT


# these params will be ignored if you flip to maxplus or maxplus-explanation-stdout
PATH_RULE_INDEX = $PATH_RULE_INDEX
MAX_EXPLANATIONS = 200

# this setting generates the explanations file and the rule index which is the input to the transformer
AGGREGATION_TYPE = maxplus-explanation

# this setting generates the ranking in the standard output of anyburl, which corresponds to the transformer input
# AGGREGATION_TYPE = maxplus-explanation-stdout

# this setting generates ranking of the original anyburl maxplus algorithm
# AGGREGATION_TYPE = maxplus

READ_CYCLIC_RULES = 1
READ_ACYCLIC1_RULES = 1
READ_ACYCLIC2_RULES = 0
READ_ZERO_RULES = 0
READ_THRESHOLD_CORRECT_PREDICTIONS = 5

TOP_K_OUTPUT = 100
WORKER_THREADS = $worker_threads
EOM

java -cp $anyburl_path de.unima.ki.anyburl.Apply $apply_config
python $process_explanations_path --data_dir $data_dir --split train --explanation_file $PATH_OUTPUT --rules_index_file $PATH_RULE_INDEX --save_dir $processed_explanations_folder


#write explanations for valid
PATH_TRAINING="${data_dir}train.txt"
PATH_VALID="${data_dir}empty.txt"
PATH_TEST="${data_dir}valid.txt"
PATH_OUTPUT="${explanations_folder}valid-explanations"
apply_config="${explanations_folder}config-apply-explanations-valid.properties"
PATH_RULE_INDEX="${explanations_folder}rule-index-valid"

/bin/cat <<EOM >$apply_config
PATH_TRAINING  = $PATH_TRAINING
PATH_VALID     = $PATH_VALID
PATH_TEST      = $PATH_TEST

PATH_RULES      = $rule_file
PATH_OUTPUT     = $PATH_OUTPUT


# these params will be ignored if you flip to maxplus or maxplus-explanation-stdout
PATH_RULE_INDEX = $PATH_RULE_INDEX
MAX_EXPLANATIONS = 200

# this setting generates the explanations file and the rule index which is the input to the transformer
AGGREGATION_TYPE = maxplus-explanation

# this setting generates the ranking in the standard output of anyburl, which corresponds to the transformer input
# AGGREGATION_TYPE = maxplus-explanation-stdout

# this setting generates ranking of the original anyburl maxplus algorithm
# AGGREGATION_TYPE = maxplus

READ_CYCLIC_RULES = 1
READ_ACYCLIC1_RULES = 1
READ_ACYCLIC2_RULES = 0
READ_ZERO_RULES = 0
READ_THRESHOLD_CORRECT_PREDICTIONS = 5

TOP_K_OUTPUT = 100
WORKER_THREADS = $worker_threads
EOM

java -cp $anyburl_path de.unima.ki.anyburl.Apply $apply_config
python $process_explanations_path --data_dir $data_dir --split valid --explanation_file $PATH_OUTPUT --rules_index_file $PATH_RULE_INDEX --save_dir $processed_explanations_folder


#write explanations for test
PATH_TRAINING="${data_dir}train.txt"
PATH_VALID="${data_dir}valid.txt"
PATH_TEST="${data_dir}test.txt"
PATH_OUTPUT="${explanations_folder}test-explanations"
apply_config="${explanations_folder}config-apply-explanations-test.properties"
PATH_RULE_INDEX="${explanations_folder}rule-index-test"

/bin/cat <<EOM >$apply_config
PATH_TRAINING  = $PATH_TRAINING
PATH_VALID     = $PATH_VALID
PATH_TEST      = $PATH_TEST

PATH_RULES      = $rule_file
PATH_OUTPUT     = $PATH_OUTPUT


# these params will be ignored if you flip to maxplus or maxplus-explanation-stdout
PATH_RULE_INDEX = $PATH_RULE_INDEX
MAX_EXPLANATIONS = 200

# this setting generates the explanations file and the rule index which is the input to the transformer
AGGREGATION_TYPE = maxplus-explanation

# this setting generates the ranking in the standard output of anyburl, which corresponds to the transformer input
# AGGREGATION_TYPE = maxplus-explanation-stdout

# this setting generates ranking of the original anyburl maxplus algorithm
# AGGREGATION_TYPE = maxplus

READ_CYCLIC_RULES = 1
READ_ACYCLIC1_RULES = 1
READ_ACYCLIC2_RULES = 0
READ_ZERO_RULES = 0
READ_THRESHOLD_CORRECT_PREDICTIONS = 5

TOP_K_OUTPUT = 100
WORKER_THREADS = $worker_threads
EOM

java -cp $anyburl_path de.unima.ki.anyburl.Apply $apply_config
python $process_explanations_path --data_dir $data_dir --split test --explanation_file $PATH_OUTPUT --rules_index_file $PATH_RULE_INDEX --save_dir $processed_explanations_folder




