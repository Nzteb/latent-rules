#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import util

if __name__ == "__main__":
    args = util.default_parser().parse_args()
    field_map = {
        "S": args.subject_field,
        "P": args.predicate_field,
        "O": args.object_field,
    }

    print(f"Preprocessing {args.folder}...")

    # register raw splits
    train_raw = util.RawSplit(
        file="train.txt",
        field_map=field_map,
        collect_entities=True,
        collect_relations=True,
    )
    
    train_raw_target = util.RawSplit(
        file="train_target.txt",
        field_map=field_map,
    )
    
    
    valid_raw = util.RawSplit(file="valid.txt", field_map=field_map,)
    test_raw = util.RawSplit(file="test.txt", field_map=field_map,)


    raw_dataset = util.analyze_raw_splits(
        raw_splits=[train_raw, train_raw_target, valid_raw, test_raw], folder=args.folder,
    )

    # create splits: TRAIN
    train = util.Split(
        raw_split=train_raw,
        key="train",
        options={"type": "triples", "filename": "train.del", "split_type": "train_context"},
    )
    
    train_target = util.Split(
        raw_split=train_raw_target,
        key="train_target",
        options={"type": "triples", "filename": "train_target.del", "split_type": "train"},
    )
   
    train_raw.splits.extend([train])
    train_raw_target.splits.extend([train_target])
    
    valid = util.Split(
            raw_split=valid_raw,
            key="valid",
            options={"type": "triples", "filename": "valid.del", "split_type": "valid"},
        )


    valid_raw.splits.extend([valid])

    test = util.Split(
            raw_split=test_raw,
            key="test",
            options={"type": "triples", "filename": "test.del", "split_type": "test"},
        )

    test_raw.splits.extend([test])


    # do the work
    util.process_splits(raw_dataset)
    util.update_string_files(raw_dataset, args)
    util.write_dataset_yaml(raw_dataset.config, args.folder)
