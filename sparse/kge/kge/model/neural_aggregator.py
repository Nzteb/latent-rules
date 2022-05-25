
import torch
from torch import Tensor
from torch import nn
import math
from torch.nn.functional import softmax
import copy

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel

class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0.15, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class NeuralAggregatorScorer(RelationalScorer):
    r"""Implementation of the DistMult KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.dataset.load_rule_meta()

        # TODO initialize as libkge does
        emb_dim = self.get_option("embedding_dim")
        dropout = self.get_option("dropout")
        self.dropout = nn.Dropout(p=dropout)
        self.emb_dim = emb_dim
        self.num_embeddings = len(self.dataset._rule_features)
        # last index holds the padding token embedding
        self.embeddings = torch.nn.Embedding(self.num_embeddings+1, emb_dim, sparse=False)
        init = self.get_option("initialize")
        if init in ["equal"]:
            self.embeddings.weight.data[:] = 1.0
        elif init in ["top_one"]:
            self.embeddings.weight.data[:, 0] = 1.0
            self.embeddings.weight.data[:, 1:] = 0.0
        elif init == "":
            pass
        else:
            self.initialize(self.embeddings.weight.data)

        self.cls_embedding = torch.nn.Embedding(1, self.get_option("embedding_dim"))

        self.final_scorer = torch.nn.Linear(emb_dim, 1)

        self.meta_sp = None
        self.meta_op = None

        self.rule_conf = None

    def score_indices(self, indices):

        # indices: max_num_rules x batch_size
        # note that first dimension can also be num_rules < max_num_rules (in score_spo)
        # e.g. not using padding tokens


        # max_num_rules x batch_size x emb_dim
        emb = self.dropout(self.embeddings(indices.long()))

        scoring_type = self.get_option("scoring_type")

        confs = torch.zeros(indices.shape).to(self.embeddings.weight.device)
        if scoring_type in ["shallow_sum", "shallow_noisy"]:
            #fetch rule confidences
            confs = self.rule_conf[indices.long()].to(self.embeddings.weight.device)

        # mask is 1 for indices to ignore
        padd_mask = torch.zeros(indices.size()[1], indices.size()[0]).to(self.embeddings.weight.device)

        # self.num_embeddings is the index of the padding embedding
        padd_mask[:, :] = (indices == self.num_embeddings).permute(1, 0)
        padd_mask = padd_mask.to(torch.bool)

        scores = self.scorer_score_emb(emb, padd_mask, confs)
        return scores

    def scorer_score_emb(self, emb, padding_mask, confs):
        """
         # emb is (cls+)num_rules x batch_size x emb_dim
        # padding mask is batch_size x num_rules
        # padding_mask[i,j] = 0 represents some rule at j fired for batch candidate i
        # padding_mask[i,j] = 1 represents padding embeddings
        confs is num_rules x batch_size (without cls)

        :return: batch_size x 1  scores/probabs
        """

        scoring_type = self.get_option("scoring_type")

        transform = emb
        if scoring_type in ["shallow_sum", "shallow_noisy"]:
            emb_dim = transform.size()[2]
            padding_mask = padding_mask.permute(1, 0)
            #batch_size x 1
            num_non_zero = (~padding_mask).sum(dim=0)
            padding_mask_rep = padding_mask.unsqueeze(2).repeat(1, 1, emb_dim)

            # this masks out all padding embeddings
            # then transform is max_num_rules(without cls)x batch_size x emb_dim
            # transform[i,j,:] = vec(0) if rule position i does not contain a rule (padding)
            transform = transform * ~padding_mask_rep

            if scoring_type in ["shallow_noisy", "shallow_sum"]:
                transform = softmax(transform, dim=2)
                transform = transform * confs.unsqueeze(2)
                # ensure none of padding embedding entries is picked in max by accident
                transform = transform.masked_fill(padding_mask_rep, float("-inf"))
                # max over rules
                #batch_size x emb_dimm
                max_transform, max_idx = transform.max(dim=0)
                prob = max_transform
                if scoring_type in ["shallow_sum"]:
                    return prob.sum(dim=1).unsqueeze(1)
                elif scoring_type in ["shallow_noisy"]:
                    prob = 1 - prob
                    product = prob.prod(dim=1)
                    return (1 - product).unsqueeze(1)
            if scoring_type == "noisy_or":
                logit = self.final_scorer.forward(transform)
                # mask padding embeddings s.t. their sigmoid value will be 0
                logit = logit.masked_fill(padding_mask.unsqueeze(2), float("-inf"))
                prob = torch.sigmoid(logit)
                prob = 1 - prob
                product = prob.prod(dim=0)
                return 1 - product

            elif scoring_type in ["pool", "no_transform"]:
                # batch_size x emb_dim
                transform = transform.sum(dim=0)
                # mean
                transform = transform / num_non_zero.view(-1, 1)
                return self.final_scorer.forward(transform)
        else:
            raise Exception(f"Wrong scoring_type: {scoring_type}")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        """Expects indices instead of embedding.
            Score_emb inferface is maintained s.t. kge_model dose not need to be overridden

        """
        # combine is "spo" and direction is "o"
        if combine == "o":
            meta_sp = self.meta_sp
            out = torch.empty(len(s_emb)).fill_(float("-1000")).to(s_emb.device)
            for i in range(len(s_emb)):
                s, p, o = s_emb[i].item(), p_emb[i].item(), o_emb[i].item()
                if not (s, p) in meta_sp:
                    continue
                candidates = meta_sp[s, p]["candidates"]
                rules = meta_sp[s, p]["rules"]
                if not o in candidates:
                    continue
                rules = rules[candidates.index(o)]
                out[i] = self.score_indices(
                    torch.tensor(rules)[:self.get_option("max_num_rules")].view(-1, 1).to(out.device)
                )
            return out

        # combine is "spo" and direction is "s"
        elif combine == "s":
            meta_po = self.meta_po
            out = torch.empty(len(o_emb)).fill_(float("-1000")).to(o_emb.device)
            for i in range(len(s_emb)):
                s, p, o = s_emb[i].item(), p_emb[i].item(), o_emb[i].item()
                if not (p, o) in meta_po:
                    continue
                candidates = meta_po[p, o]["candidates"]
                rules = meta_po[p, o]["rules"]
                if not s in candidates:
                    continue
                rules = rules[candidates.index(s)]
                out[i] = self.score_indices(
                    torch.tensor(rules)[:self.get_option("max_num_rules")].view(-1, 1).to(out.device)
                )
            return out

        elif combine == "sp_":
            max_num_rules = self.get_option("max_num_rules")
            meta_sp = self.meta_sp
            out = torch.empty(len(s_emb), len(o_emb)).fill_(float("-1000")).to(s_emb.device)
            for i in range(len(s_emb)):
                s, p, = s_emb[i].item(), p_emb[i].item()
                if (s, p) not in meta_sp:
                    continue

                # all candidates for which some rules fired
                candidates = meta_sp[s, p]["candidates"]
                rules = meta_sp[s, p]["rules"]
                num_cand = len(candidates)
                if not num_cand:
                    continue
                # find matches of o_emb (candidates to score) and candidates (candidates for which rules fired)
                # coords has variable length and coords[k] = [i,j] with candidates[i] == o_emb[j]
                coords = (torch.tensor(candidates).unsqueeze(1).to(o_emb.device) == o_emb).nonzero(as_tuple=False)
                batch_indices = torch.empty(
                    max_num_rules,
                    len(coords),
                ).fill_(self.num_embeddings).to(s_emb.device)  # index of the padding token embedding
                batch_idx = 0
                for coord in coords:
                    idx_candidate = coord[0]
                    candidate_rules = rules[idx_candidate]
                    num_rules = len(candidate_rules)
                    num_rules = (max_num_rules if (num_rules > max_num_rules) else num_rules)
                    batch_indices[:num_rules, batch_idx] = torch.tensor(candidate_rules)[:num_rules]
                    batch_idx += 1

                out[i, coords[:, 1]] = self.score_indices(batch_indices).flatten()
            return out

        elif combine == "_po":
            max_num_rules = self.get_option("max_num_rules")
            meta_po = self.meta_po
            out = torch.empty(len(o_emb), len(s_emb)).fill_(float("-1000")).to(s_emb.device)
            for i in range(len(o_emb)):
                p, o = p_emb[i].item(), o_emb[i].item()
                if (p, o) not in meta_po:
                    continue
                candidates = meta_po[p, o]["candidates"]
                rules = meta_po[p, o]["rules"]
                num_cand = len(candidates)
                if not num_cand:
                    continue
                # find matches of o_emb (candidates to score) and candidates (candidates for which rules fired)
                # coords has variable length and coords[k] = [i,j] with candidates[i] == s_emb[j]
                coords = (torch.tensor(candidates).unsqueeze(1).to(s_emb.device) == s_emb).nonzero(as_tuple=False)
                batch_indices = torch.empty(
                    max_num_rules,
                    len(coords),
                ).fill_(self.num_embeddings).to(s_emb.device)  # index of the padding token embedding

                batch_idx = 0
                for coord in coords:
                    idx_candidate = coord[0]
                    candidate_rules = rules[idx_candidate]
                    num_rules = len(candidate_rules)
                    num_rules = (max_num_rules if (num_rules > max_num_rules) else num_rules)
                    batch_indices[:num_rules, batch_idx] = torch.tensor(candidate_rules)[:num_rules]
                    batch_idx += 1

                out[i, coords[:, 1]] = self.score_indices(batch_indices).flatten()
            return out
        else:
            raise Exception


class NeuralAggregator(KgeModel):
    r"""Implementation of the DistMult KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=NeuralAggregatorScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

        self._scorer.rule_conf = copy.deepcopy(dataset.rule_conf)
        del dataset.rule_conf

        self.num_embeddings = self._scorer.num_embeddings

        self._scorer.meta_sp = self.dataset._cand_sp_valid
        self._scorer.meta_po = self.dataset._cand_po_valid

        self._entity_embedder.scorer = self.get_scorer()
        self._relation_embedder.scorer = self.get_scorer()

    def score_indices(self, indices):
        return self._scorer.score_indices(indices)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        if direction == "o":
            return self._scorer.score_emb(s, p, o, combine="o")
        elif direction == "s":
            return self._scorer.score_emb(s, p, o, combine="s")
        else:
            raise Exception("No direction")

    def penalty(self, **kwargs):
        return []

