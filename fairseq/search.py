# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn.functional as F

from fairseq.gumbel import gumbel_like, gumbel_with_maximum
from fairseq.utils import log_add

import numpy as np
from fairseq.cps_dp import sample
from torch.multiprocessing import Pool
from torch import multiprocessing
import cProfile


class Search(object):

    def __init__(self, tgt_dict):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.log_ps_buf = None
        self.log_ps_t_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.log_ps_buf = t.new()
            self.log_ps_t_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores, beam_size):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


class CPS(Search):
    def __init__(self, tgt_dict, sampling_topk=-1, sampling_temperature=1.0):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        assert self.sampling_topk == -1
        self.sampling_temperature = sampling_temperature

    def _init_buffers(self, t):
        super()._init_buffers(t)

        self.remaining_subsetsum_product_probs = None
        self.subset_sum_product_probs = None
        self.dp = None
        self.p = None
        self.samples_idx = torch.LongTensor().to(device=t.device)
        self.log_inclusion_probs = torch.FloatTensor().to(device=t.device)

    def _calc_inclusion_probs(self, p, k, j):
        n = len(p)
        self.dp = np.zeros((n + 1))
        self.remaining_subsetsum_product_probs = np.zeros((k + 2, n + 2))
        self.remaining_subsetsum_product_probs[k, :] = 1
        for r in range(k, 0, -1):
            for i in range(n, 0, -1):
                self.dp[i] += self.subset_sum_product_probs[r - 1, i - 1] * self.remaining_subsetsum_product_probs[r, i + 1]
                self.remaining_subsetsum_product_probs[r, i] = self.remaining_subsetsum_product_probs[r + 1, i + 1] * p[i] + \
                                                          self.remaining_subsetsum_product_probs[r, i + 1]
        p_cliped = p[1:]
        dp_cliped = self.dp[1:]
        inclusion_probs = p_cliped * dp_cliped / self.subset_sum_product_probs[k, n]
        return inclusion_probs

    def cps_sample(self, logp, k, bsz):
        n = logp.size()[1]
        torch.zeros([bsz, k], dtype=torch.int64, out=self.samples_idx)
        torch.zeros([bsz, n], out=self.log_inclusion_probs)

        logp_np = logp.detach().numpy()
        logp_np = logp_np.astype(np.float64)

        # print("I am using {} number of cores".format(multiprocessing.cpu_count()))
        # with Pool(processes=multiprocessing.cpu_count()) as pool:
        #     multiple_results = [pool.apply_async(sample, args=(logp_np[j,:], k, bsz)) for j in range(bsz)]
        #     sample_idx_np = np.asarray([el.get() for el in multiple_results])

        for j in range(bsz):
            sample_idx_np, log_inc_np = sample(logp_np[j,:], k, bsz)
            self.samples_idx[j] = torch.from_numpy(sample_idx_np)
            self.log_inclusion_probs[j] = torch.from_numpy(log_inc_np)

        return self.log_inclusion_probs, self.samples_idx

    def step(self, step, lprobs, scores, log_ps, log_ps_t):
        bsz, beam_size, vocab_size = lprobs.size()
        self._init_buffers(lprobs)

        lprobs_t = lprobs.clone()
        if self.sampling_temperature != 1.0:
            lprobs_t = F.log_softmax(lprobs / self.sampling_temperature, -1)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs_t = lprobs_t[:, ::beam_size, :].contiguous()
            lprobs = lprobs[:, ::beam_size, :].contiguous()

            self.log_ps_buf = lprobs.clone()
            self.log_ps_t_buf = lprobs_t.clone()

            # make probs contain cumulative scores for each hypothesis
            # lprobs_t.add_(log_ps_t[:, :, step - 1].unsqueeze(-1))
            # lprobs.add_(log_ps[:, :, step - 1].unsqueeze(-1))

        else:
            self.log_ps_buf = torch.add(lprobs, log_ps[:, :, step - 1].unsqueeze(-1))
            self.log_ps_t_buf = torch.add(lprobs_t, log_ps_t[:, :, step - 1].unsqueeze(-1))

        cand_scores, self.indices_buf = self.cps_sample(lprobs_t.view(bsz, -1), beam_size, bsz)

        # Gather cumulative
        log_ps_buf_buf = self.log_ps_buf.clone()
        log_ps_t_buf_buf = self.log_ps_t_buf.clone()
        #
        if step != 0:
            cand_scores = torch.reshape(cand_scores, (bsz, beam_size, -1))
            cand_scores.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.gather(
            cand_scores.view(bsz, -1), -1, self.indices_buf, out=self.scores_buf
        )
        print(self.scores_buf)

        torch.gather(
            log_ps_buf_buf.view(bsz, -1), -1, self.indices_buf, out=self.log_ps_buf
        )

        torch.gather(
            log_ps_t_buf_buf.view(bsz, -1), -1, self.indices_buf, out=self.log_ps_t_buf
        )

        torch.floor_divide(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.log_ps_buf, self.log_ps_t_buf, self.indices_buf, self.beams_buf


class BeamSearch(Search):

    def __init__(self, tgt_dict, naive_stochastic=False, stochastic=False, sampling_topk=-1, sampling_temperature=1.0):
        super().__init__(tgt_dict)
        self.stochastic = stochastic
        self.naive_stochastic = naive_stochastic
        self.sampling_topk = sampling_topk
        assert self.sampling_topk == -1, "Sampling top-k for beam search not yet supported"
        self.sampling_temperature = sampling_temperature

    def step(self, step, lprobs, scores, log_ps, log_ps_t):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        lprobs_t = lprobs.clone()
        if self.sampling_temperature != 1.0:
            lprobs_t = F.log_softmax(lprobs / self.sampling_temperature, -1)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs_t = lprobs_t[:, ::beam_size, :].contiguous()
            lprobs = lprobs[:, ::beam_size, :].contiguous()

            if self.stochastic or self.naive_stochastic:
                cand_scores = gumbel_like(lprobs_t) + lprobs_t
            else:
                cand_scores = lprobs_t
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs_t.add_(log_ps_t[:, :, step - 1].unsqueeze(-1))
            lprobs.add_(log_ps[:, :, step - 1].unsqueeze(-1))

            if self.stochastic:
                assert self.sampling_topk == -1
                cand_scores, _ = gumbel_with_maximum(lprobs_t, scores[:, :, step - 1], -1)
            else:
                cand_scores = lprobs_t

        torch.topk(
            cand_scores.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                cand_scores.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )

        # Gather cumulative
        torch.gather(
            lprobs.view(bsz, -1), -1, self.indices_buf, out=self.log_ps_buf
        )

        if self.stochastic or self.naive_stochastic:
            torch.gather(
                lprobs_t.view(bsz, -1), -1, self.indices_buf, out=self.log_ps_t_buf
            )
        else:
            self.log_ps_t_buf = self.scores_buf

        torch.floor_divide(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.log_ps_buf, self.log_ps_t_buf, self.indices_buf, self.beams_buf


class LengthConstrainedBeamSearch(Search):

    def __init__(self, tgt_dict, min_len_a, min_len_b, max_len_a, max_len_b):
        super().__init__(tgt_dict)
        self.min_len_a = min_len_a
        self.min_len_b = min_len_b
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = BeamSearch(tgt_dict)

    def step(self, step, lprobs, scores):
        min_lens = self.min_len_a * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step == max_lens, :, self.eos] = 0
        lprobs[step > max_lens, :, self.eos] = -math.inf
        return self.beam.step(step, lprobs, scores)


class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.diversity_buf = None
        self.beam = BeamSearch(tgt_dict)

    def step(self, step, lprobs, scores, log_ps, log_ps_t):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                'DiverseBeamSearch requires --beam to be divisible by the number of groups'
            )

        # initialize diversity penalty
        if self.diversity_buf is None:
            self.diversity_buf = lprobs.new()
        torch.zeros(lprobs[:, 0, :].size(), out=self.diversity_buf)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g::self.num_groups, :]
            scores_g = scores[:, g::self.num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(lprobs_g, self.diversity_strength, self.diversity_buf.unsqueeze(1))
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, _, _, indices_buf, beams_buf = self.beam.step(step, lprobs_g, scores_g, scores_g, scores_g)
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            self.diversity_buf.scatter_add_(
                1,
                indices_buf,
                self.diversity_buf.new_ones(indices_buf.size())
            )

        # interleave results from different groups
        self.scores_buf = torch.stack(scores_G, dim=2, out=self.scores_buf).view(bsz, -1)
        self.indices_buf = torch.stack(indices_G, dim=2, out=self.indices_buf).view(bsz, -1)
        self.beams_buf = torch.stack(beams_G, dim=2, out=self.beams_buf).view(bsz, -1)
        return self.scores_buf, self.scores_buf, self.scores_buf, self.indices_buf, self.beams_buf


class Sampling(Search):

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_temperature=1.):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_temperature = sampling_temperature

    def step(self, step, lprobs, scores, log_ps, log_ps_t):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        # we exclude the first two vocab items, one of which is pad
        assert self.pad <= 1, 'sampling assumes the first two symbols can be ignored'
        lprobs_nopad = lprobs[:, :, 2:]

        # only sample from top-k candidates
        if self.sampling_topk > 0:
            lprobs_nopad, topk_indices = lprobs_nopad.topk(self.sampling_topk)

        # sampling temperature
        if self.sampling_temperature != 1.:
            lprobs_nopad_t = F.log_softmax(lprobs_nopad / self.sampling_temperature, -1)
        else:
            lprobs_nopad_t = lprobs_nopad

        # sample
        probs_nopad_t = lprobs_nopad_t.exp()
        if step == 0:
            self.indices_buf = torch.multinomial(
                probs_nopad_t.view(bsz, -1),
                beam_size,
                replacement=True,
                out=self.indices_buf,
            ).view(bsz, beam_size)
        else:
            self.indices_buf = torch.multinomial(
                probs_nopad_t.view(bsz * beam_size, -1),
                1,
                replacement=True,
                out=self.indices_buf,
            ).view(bsz, beam_size)

        if step == 0:
            # expand to beam size
            lprobs_nopad = lprobs_nopad.expand(bsz, beam_size, -1)
            lprobs_nopad_t = lprobs_nopad_t.expand(bsz, beam_size, -1)

        # gather probs
        torch.gather(
            lprobs_nopad,
            dim=2,
            index=self.indices_buf.unsqueeze(-1),
            out=self.log_ps_buf,
        )
        torch.gather(
            lprobs_nopad_t,
            dim=2,
            index=self.indices_buf.unsqueeze(-1),
            out=self.log_ps_t_buf,
        )
        self.log_ps_buf = self.log_ps_buf.view(bsz, -1)
        self.log_ps_t_buf = self.log_ps_t_buf.view(bsz, -1)

        # remap indices if using top-k sampling
        if self.sampling_topk > 0:
            self.indices_buf = torch.gather(
                topk_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=self.indices_buf.unsqueeze(-1),
            ).squeeze(2)

        # remap indices since we excluded the first two vocab items
        self.indices_buf.add_(2)

        if step == 0:
            self.beams_buf = self.indices_buf.new_zeros(bsz, beam_size)
        else:
            self.beams_buf = torch.arange(0, beam_size, out=self.beams_buf).repeat(bsz, 1)
            # make log_ps cumulative
            self.log_ps_buf.add_(
                torch.gather(
                    log_ps[:, :, step - 1],
                    dim=1,
                    index=self.beams_buf,
                )
            )
            # make log_ps_t cumulative
            self.log_ps_t_buf.add_(
                torch.gather(
                    log_ps_t[:, :, step - 1],
                    dim=1,
                    index=self.beams_buf,
                )
            )
        # Scores buf is not used for sampling
        self.scores_buf = self.log_ps_buf.clone()
        return self.scores_buf, self.log_ps_buf, self.log_ps_t_buf, self.indices_buf, self.beams_buf
