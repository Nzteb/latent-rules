import ast
import copy
import os.path
from os import path
import pickle
import argparse


def preprocess_rules(file, relation_ids):
    # key is relation idx value is a list where the index is local index and entry is global index
    rule_map = {}
    # key is global rule index value is a list where first entry is #the rules fired and second entry is # fired correctly
    rule_features = {}
    with open(file, "r") as f:
        txt = f.read()
        txt = txt.split(">>>")[1:]

        global_index = 0
        for relation in txt:
            raw = relation.split("\n")
            rules_raw = raw[1:]
            rel_raw = raw[0]
            rule_count = 0
            rel_idx = relation_ids.index(rel_raw.strip())
            rule_map[rel_idx] = []
            if not rules_raw[-1] == "":
                raise Exception("Sanity check failed.")
            for rule in rules_raw[:-1]:
                rule_line = rule.split("\t")
                local_rule_index = int(rule_line[0])
                if local_rule_index != rule_count:
                    raise Exception("Rule indices are not sorted for a relation")
                rule_map[rel_idx].append(global_index)
                rule_features[global_index] = [rule_line[1], rule_line[2], rule_line[4]]
                global_index += 1
                rule_count += 1
        return rule_map, rule_features


def preprocess_candidates(file: str, entity_ids: list, relation_ids: list, rule_map: dict):
    processed = {}
    processed_sp = {}
    processed_po = {}

    longest = 0
    cnt = 0
    with open(file, "r") as f:
        txt = f.read().split("}}},")

        if txt[-1] == "\n":
            txt = txt[:-1]

        for line in txt:
            cnt += 1
            if cnt % 100 ==0:
                print(f"processed {cnt} explanations")
            proc = line.replace("\n", "") + "}}}"
            raw_dict = ast.literal_eval(proc)
            raw_triple = list(raw_dict.keys())[0].split(" ")
            S, P, O = entity_ids.index(raw_triple[0]), relation_ids.index(raw_triple[1]), entity_ids.index(
                raw_triple[2])
            raw_meta = list(raw_dict.values())[0]
            raw_heads = raw_meta["heads"]
            raw_tails = raw_meta["tails"]

            # reflexive predictions in anyburl are represented like this
            # e.g. use the tail of the triple for the head completion
            if "me_myself_i" in raw_heads["candidates"]:
                raw_heads["candidates"][raw_heads["candidates"].index("me_myself_i")] = raw_triple[2]

            if "me_myself_i" in raw_tails["candidates"]:
                raw_tails["candidates"][raw_tails["candidates"].index("me_myself_i")] = raw_triple[0]

            raw_heads["candidates"] = list(map(lambda can: entity_ids.index(can), raw_heads["candidates"]))
            raw_tails["candidates"] = list(map(lambda can: entity_ids.index(can), raw_tails["candidates"]))

            rules_heads = []
            # assign global rule indices to the rules
            for candidate in raw_heads["rules"]:
                cands = []
                for local_rule_index in candidate:
                    if P in rule_map:
                        global_idx = rule_map[P][local_rule_index]
                        cands.append(global_idx)
                if len(cands) > longest:
                    longest = len(cands)
                rules_heads.append(cands)
            raw_heads["rules"] = rules_heads

            rules_tails = []
            # assign global rule indices to the rules
            for candidate in raw_tails["rules"]:
                cands = []
                for local_rule_index in candidate:
                    if P in rule_map:
                        global_idx = rule_map[P][local_rule_index]
                        cands.append(global_idx)
                rules_tails.append(cands)
                if len(cands) > longest:
                    longest = len(cands)
            raw_tails["rules"] = rules_tails

            # raw meta has been processed via raw_tails and raw_heads
            processed[(S, P, O)] = raw_meta

            if (S, P) in processed_sp:
                # add the candidate to the sp_ candidates if it is within the topk
                if O in raw_meta["tails"]["candidates"]:
                    processed_sp[(S, P)]["candidates"].append(O)
                    idx_O = raw_meta["tails"]["candidates"].index(O)
                    processed_sp[(S, P)]["rules"].append(copy.copy(raw_meta["tails"]["rules"][idx_O]))
            else:
                processed_sp[(S, P)] = {}
                processed_sp[(S, P)]["candidates"] = copy.deepcopy(raw_meta["tails"]["candidates"])
                processed_sp[(S, P)]["rules"] = copy.deepcopy(raw_meta["tails"]["rules"])

            if (P, O) in processed_po:
                if S in raw_meta["heads"]["candidates"]:
                    processed_po[(P, O)]["candidates"].append(S)
                    idx_S = raw_meta["heads"]["candidates"].index(S)
                    processed_po[(P, O)]["rules"].append(copy.copy(raw_meta["heads"]["rules"][idx_S]))
            else:
                processed_po[(P, O)] = {}
                processed_po[(P, O)]["candidates"] = copy.deepcopy(raw_meta["heads"]["candidates"])
                processed_po[(P, O)]["rules"] = copy.deepcopy(raw_meta["heads"]["rules"])

        print(f"Longest rule set for a candidate: {longest}")
        return processed, processed_sp, processed_po


def read_ids(file):
    with open(file, "r") as f:
        raw = f.read().splitlines()
    ids = []
    for ent in raw:
        ids.append(ent.split("\t")[1])
    return ids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--explanation_file", required=True)
    parser.add_argument("--rules_index_file", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    data_dir = args.data_dir
    split = args.split
    explanation_file = args.explanation_file
    rules_index_file = args.rules_index_file
    save_dir = args.save_dir
    if split not in ["train", "valid", "test"]:
        raise Exception
    print(f"Processing {split}")


    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)



    ent_ids_file = os.path.join(data_dir, "entity_ids.del")
    entity_ids = read_ids(ent_ids_file)
    rel_ids_file = os.path.join(data_dir, "relation_ids.del")
    relation_ids = read_ids(rel_ids_file)

    # rule_map is a dict where key is relation id, values are lists a where a[local_rule_index_for_relation] = global_rule_index
    # rule features is dict where key is the global rule index value is a list where first el denotes number the
    # rule fired, second el is number the rule fired correctly and third is the rule string representation
    rule_map, rule_features = preprocess_rules(file=rules_index_file, relation_ids=relation_ids)

    processed_candidates, processed_sp, processed_po = preprocess_candidates(
        explanation_file,
        entity_ids=entity_ids,
        relation_ids=relation_ids,
        rule_map=rule_map
    )

    print(f"Found rules for {len(rule_map)} relations.")

    save_path = os.path.join(save_dir, f"processed_explanations_{split}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(processed_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(save_dir, f"processed_sp_{split}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(processed_sp, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(save_dir, f"processed_po_{split}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(processed_po, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(save_dir, "rule_map.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(rule_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(save_dir, "rule_features.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(rule_features, f, protocol=pickle.HIGHEST_PROTOCOL)
