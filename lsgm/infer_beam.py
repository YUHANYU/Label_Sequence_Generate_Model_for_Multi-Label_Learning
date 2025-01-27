"""Class for generating sequences
Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import heapq
from .config import so


class Sequence(object):
    """Represents a complete or partial sequence. 表示完整或部分的序列"""

    def __init__(self, output, state, logprob, score, attention=None):
        """Initializes the Sequence.
        Args:
          output: List of word ids in the sequence. 序列中的单词id列表
          state: Model state after generating the previous word. 生成前一个单词后的模型状态
          logprob: Log-probability of the sequence. 序列的对数概率
          score: Score of the sequence. 序列的得分
        """
        self.output = output
        self.state = state
        self.logprob = logprob
        self.score = score
        self.attention = attention

    def __cmp__(self, other):
        """Compares Sequences by score. 按分数比较序列"""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set.保持提供集合递增性的前n个元素。"""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 decode_step,
                 eos_id=EOS,
                 beam_size=3,
                 max_sequence_length=50,
                 get_attention=False,
                 length_normalization_factor=0.0,
                 length_normalization_const=5.,
                 device_ids=None):
        """Initializes the generator.
        Args:
          deocde_step: function, with inputs: (input, state) and outputs len(vocab) values
          eos_id: the token number symobling the end of sequence
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.decode_step = decode_step
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.get_attention = get_attention
        self.device_ids = device_ids

    def beam_search(self, initial_input, initial_state=None):
        """Runs beam search sequence generation on a single image.
        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
          initial_state (optional): An initial state for the model -
                         list of batch size holding the current state for every entry.
        Returns:
          A list of batch size, each the most likely sequence from the possible beam_size candidates.
        """
        batch_size = len(initial_input)
        partial_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN(self.beam_size) for _ in range(batch_size)]

        words, logprobs, new_state = self.decode_step(
            initial_input, initial_state,
            k=self.beam_size,
            feed_all_timesteps=True,
            get_attention=self.get_attention)
        for b in range(batch_size):
            # Create first beam_size candidate hypotheses for each entry in
            # batch
            for k in range(self.beam_size):
                seq = Sequence(
                    output=initial_input[b] + [words[b][k]],
                    state=new_state[b],
                    logprob=logprobs[b][k],
                    score=logprobs[b][k],
                    attention=None if not self.get_attention else [new_state[b].attention_score])
                partial_sequences[b].push(seq)

        # Run beam search.
        for _ in range(self.max_sequence_length - 1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()

            # Keep a flattened list of parial hypotheses, to easily feed
            # through a model as whole batch
            flattened_partial = [
                s for sub_partial in partial_sequences_list for s in sub_partial]

            input_feed = [c.output for c in flattened_partial]
            state_feed = [c.state for c in flattened_partial]
            if len(input_feed) == 0:
                # We have run out of partial candidates; happens when
                # beam_size=1
                break

            # Feed current hypotheses through the model, and recieve new outputs and states
            # logprobs are needed to rank hypotheses
            words, logprobs, new_states \
                = self.decode_step(
                    input_feed, state_feed,
                    k=self.beam_size + 1,
                    get_attention=self.get_attention,
                    device_ids=self.device_ids)

            idx = 0
            for b in range(batch_size):
                # For every entry in batch, find and trim to the most likely
                # beam_size hypotheses
                for partial in partial_sequences_list[b]:
                    state = new_states[idx]
                    if self.get_attention:
                        attention = partial.attention + \
                            [new_states[idx].attention_score]
                    else:
                        attention = None
                    k = 0
                    num_hyp = 0
                    while num_hyp < self.beam_size:
                        w = words[idx][k]
                        output = partial.output + [w]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob
                        k += 1
                        num_hyp += 1

                        if w.item() == self.eos_id:
                            if self.length_normalization_factor > 0:
                                L = self.length_normalization_const
                                length_penalty = (L + len(output)) / (L + 1)
                                score /= length_penalty ** self.length_normalization_factor
                            beam = Sequence(output, state,
                                            logprob, score, attention)
                            complete_sequences[b].push(beam)
                            num_hyp -= 1  # we can fit another hypotheses as this one is over
                        else:
                            beam = Sequence(output, state,
                                            logprob, score, attention)
                            partial_sequences[b].push(beam)
                    idx += 1

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.
        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]
        seqs = [complete.extract(sort=True)[0]
                for complete in complete_sequences]
        return seqs