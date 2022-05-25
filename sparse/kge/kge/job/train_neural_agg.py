import time
import copy
import random

import torch
import torch.utils.data

import numpy as np

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.indexing import where_in


class TrainingNeuralAggregation(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll-NeuralAggregation"
        self.max_neg = self.config.get("NeuralAggregation.max_neg_candidates")

        if self.__class__ == TrainingNeuralAggregation:
            for f in Job.job_created_hooks:
                f(self)

    def _get_collate(self):

        # pre-compute s.t. workers dont compute it individually
        self.dataset.index("train_po_to_s")
        self.dataset.index("train_sp_to_o")

        def collate(batch):
            measure = 0
            measure -= time.time()
            triples = self.dataset.split(self.train_split)[batch, :].long()

            # list of tensors with input indices
            sp_input = []
            # list with the according index of the true candidate
            sp_labels = []

            po_input = []
            po_labels = []

            max_num_rules = self.model.get_option("max_num_rules")

            for triple in triples:
                sp = triple[0].item(), triple[1].item()
                tails_candidates = copy.deepcopy(self.dataset._cand_sp_train[sp]["candidates"])
                tails_rules = copy.deepcopy(self.dataset._cand_sp_train[sp]["rules"])

                num_cand = len(tails_candidates)
                # candidates in tail direction exist and the true candidate is included
                if num_cand and triple[2] in tails_candidates:
                    idx_true = tails_candidates.index(triple[2])
                    _tails_candidates = []
                    _tails_rules = []
                    # removes the target tail from tails_candidates
                    _tails_candidates.append(tails_candidates.pop(idx_true))
                    _tails_rules.append(tails_rules.pop(idx_true))

                    # filter non target but true o's out
                    true_os = self.dataset.index("train_sp_to_o")[sp]
                    not_in = where_in(np.array(tails_candidates), true_os.numpy(), not_in=True)

                    [
                        (_tails_rules.append(tails_rules[idx]), _tails_candidates.append(tails_candidates[idx]))
                        for idx in not_in
                    ]

                    tails_rules = _tails_rules
                    tails_candidates = _tails_candidates
                    num_cand = len(tails_candidates)

                    # sentence_length x batch_size
                    # (here) max_num_rules x num_candidates
                    batch_indices = torch.empty(
                        max_num_rules,
                        num_cand,
                    ).fill_(self.model.num_embeddings)  # index of the padding token embedding
                    break_ = False
                    for batch_idx, candidate_rules in enumerate(tails_rules):
                        num_rules = len(candidate_rules)
                        # skip query if true cand. has no explanations
                        if num_rules == 0:
                            break_ = True
                            break
                        num_rules = (max_num_rules if (num_rules > max_num_rules) else num_rules)
                        batch_indices[:num_rules, batch_idx] = torch.tensor(candidate_rules[:num_rules])
                    if break_:
                        continue
                    sp_input.append(copy.deepcopy(batch_indices))
                    sp_labels.append(copy.deepcopy(tails_candidates.index(triple[2])))

                ## obtain batch tensors for candidates of po
                po = triple[1].item(), triple[2].item()
                heads_candidates = copy.deepcopy(self.dataset._cand_po_train[po]["candidates"])
                heads_rules = copy.deepcopy(self.dataset._cand_po_train[po]["rules"])

                num_cand = len(heads_candidates)
                if num_cand and triple[0] in heads_candidates:
                    idx_true = heads_candidates.index(triple[0])
                    _heads_candidates = []
                    _heads_rules = []
                    _heads_candidates.append(heads_candidates.pop(idx_true))
                    _heads_rules.append(heads_rules.pop(idx_true))

                    # filter non target but true s's out
                    true_ss = self.dataset.index("train_po_to_s")[po]
                    not_in = where_in(np.array(heads_candidates), true_ss.numpy(), not_in=True)

                    [
                        (_heads_rules.append(heads_rules[idx]), _heads_candidates.append(heads_candidates[idx]))
                        for idx in not_in
                    ]
                    heads_rules = _heads_rules
                    heads_candidates = _heads_candidates
                    num_cand = len(heads_candidates)
                    # sentence_length x batch_size
                    # (here) max_num_rules x num_candidates
                    batch_indices = torch.empty(
                        max_num_rules,
                        num_cand,
                    ).fill_(self.model.num_embeddings)  # index of the padding token embedding
                    break_ = False
                    for batch_idx, candidate_rules in enumerate(heads_rules):
                        num_rules = len(candidate_rules)
                        # skip query when true cand. has no explanations
                        if num_rules == 0:
                            break_ = True
                            break
                        num_rules = (max_num_rules if (num_rules > max_num_rules) else num_rules)
                        batch_indices[:num_rules, batch_idx] = torch.tensor(candidate_rules[:num_rules])
                    if break_:
                        continue
                    po_input.append(copy.deepcopy(batch_indices))
                    po_labels.append(copy.deepcopy(heads_candidates.index(triple[0])))

            measure += time.time()
            batch_size = len(triples)

            return {
                "sp_input": sp_input,
                "sp_labels": sp_labels,
                "po_input": po_input,
                "po_labels": po_labels,
                "batch_size": batch_size,
            }

        return collate

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.dataset.load_rule_meta()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = batch["batch_size"]

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):

        # debug
        # self.is_forward_only = True
        # result.avg_loss = 0
        # return

        batch_size = result.size
        result.prepare_time -= time.time()
        sp_input = batch["sp_input"]
        sp_labels = batch["sp_labels"]
        po_input = batch["po_input"]
        po_labels = batch["po_labels"]

        result.prepare_time += time.time()
        loss_value_sp = 0
        for idx, indices_sp in enumerate(sp_input):
            result.forward_time -= time.time()
            scores_sp = self.model.score_indices(indices_sp.to(self.device))
            loss_value_sp += self.loss(
                scores_sp.permute(1, 0),
                torch.tensor(sp_labels[idx], device=self.device).view(1)
            )
            result.forward_time += time.time()

        if not self.is_forward_only:
            result.backward_time -= time.time()
            loss_value_sp = loss_value_sp
            result.avg_loss += loss_value_sp.item()
            loss_value_sp.backward()
            result.backward_time += time.time()

        loss_value_po = 0
        for idx, indices_po in enumerate(po_input):
            result.forward_time -= time.time()
            scores_po = self.model.score_indices(indices_po.to(self.device))
            loss_value_po += self.loss(
                scores_po.permute(1, 0),
                torch.tensor(po_labels[idx], device=self.device).view(1)
                )
            result.forward_time += time.time()
        if not self.is_forward_only:
            if not type(loss_value_po) == int:
                result.backward_time -= time.time()
                loss_value_po = loss_value_po
                result.avg_loss += loss_value_po.item()
                result.avg_loss = result.avg_loss / (2 * batch_size)
                loss_value_po.backward()
                result.backward_time += time.time()
            else:
                pass

        print(f"forward: {result.forward_time}")
        print(f"backward: {result.backward_time}")
        print(f"prepare: {result.prepare_time}")


