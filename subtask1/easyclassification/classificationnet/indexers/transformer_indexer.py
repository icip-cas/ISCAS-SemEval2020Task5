# -*- coding: utf-8 -*-
import logging
from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from transformers import GPT2Tokenizer
from transformers.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("cip_transformer_indexer")
class TransformerIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` uses a tokenizer from the ``pytorch_transformers`` repository to
    index tokens.  This ``Indexer`` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``pytorch_transformers`` model to use.
    do_lowercase : ``str``
        Whether to lowercase the tokens (this should match the casing of the model name that you
        pass)
    namespace : ``str``, optional (default=``tags``)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of ``tags`` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    """

    # pylint: disable=no-self-use
    def __init__(self,
                 model_name: str,
                 do_lowercase: bool,
                 namespace: str = "tags",
                 token_min_padding_length: int = 0,
                 max_pieces: int = 512,
                 use_starting_offsets: bool = True,
                 truncate_long_sequences: bool = True,
                 start_sub_words: List[str] = None,
                 end_sub_words: List[str] = None,
                 separator_sub_word: str = "[SEP]",
                 never_lowercase: List[str] = None
                 ) -> None:
        super().__init__(token_min_padding_length)
        if model_name.endswith("-cased") and do_lowercase:
            logger.warning("Your pretrained model appears to be cased, "
                           "but your indexers is lowercasing tokens.")
        elif model_name.endswith("-uncased") and not do_lowercase:
            logger.warning("Your pretrained model appears to be uncased, "
                           "but your indexers is not lowercasing tokens.")
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lowercase)
        self._namespace = namespace
        self._added_to_vocabulary = False
        self._padding_value = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        logger.info(f"Using token indexers padding value of {self._padding_value}")

        self._never_lowercase = never_lowercase or []
        self._use_starting_offsets = use_starting_offsets
        self._max_pieces = max_pieces
        self._truncate_long_sequences = truncate_long_sequences
        self._do_lowercase = do_lowercase

        if start_sub_words:
            self._start_sub_words = start_sub_words
        else:
            if 'roberta' in model_name:
                self._start_sub_words = ['<s>']
            elif 'bert' in model_name:
                self._start_sub_words = ['[CLS]']
            elif 'xlm' in model_name:
                self._start_sub_words = ['</s>']
            elif 'gpt' in model_name or 'transfo' in model_name:
                self._start_sub_words = []
            else:
                raise ValueError("strange input")

        if end_sub_words:
            self._end_sub_words = end_sub_words
        else:
            if 'roberta' in model_name or 'xlm' in model_name:
                self._end_sub_words = ['</s>']
            elif 'bert' in model_name:
                self._end_sub_words = ['[SEP]']
            elif 'gpt' in model_name or 'transfo' in model_name:
                self._end_sub_words = []
            else:
                raise ValueError("strange input")
        # sub_word padding的时候用上面的padding_value，其它项进行padding的时候，用这个
        self._other_padding_value = 0

        self._start_sub_word_ids = [self.tokenizer.convert_tokens_to_ids(sub_word)
                                    for token in (self._start_sub_words or [])
                                    for sub_word in self.tokenizer.tokenize(token)]

        self._end_sub_word_ids = [self.tokenizer.convert_tokens_to_ids(sub_word)
                                  for token in (self._end_sub_words or [])
                                  for sub_word in self.tokenizer.tokenize(token)]

        self._separator_ids = [self.tokenizer.convert_tokens_to_ids(sub_word)
                               for sub_word in self.tokenizer.tokenize(separator_sub_word)]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.tokenizer.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    def __eq__(self, other):
        if isinstance(other, TransformerIndexer):
            for key in self.__dict__:
                if key == 'tokenizer':
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented

    """
    该indexer改自allennlp/data/token_indexers/pretrained_transformer_indexer.py
    上面的跟之前的保持不变，改动代码从下面开始
    还未能支持两个句子的编码（未实现type-ids），仅对句子过长进行截断处理，而无采用滑动窗口
    """

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary and hasattr(self.tokenizer, "vocab"):
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        self._sub_word_indexer_name = index_name
        token_text = [token.text.lower() if self._do_lowercase and token.text not in self._never_lowercase
                      else token.text for token in tokens]

        if isinstance(self.tokenizer, GPT2Tokenizer):
            tokens = list()
            for index, token in enumerate(token_text):
                add_prefix_space = (index != 0)
                tokens += [self.tokenizer.tokenize(token, add_prefix_space=add_prefix_space)]
            sub_word_ids = [[self.tokenizer.convert_tokens_to_ids(sub_word) for sub_word in token] for token in tokens]
        else:
            sub_word_ids = [
                [self.tokenizer.convert_tokens_to_ids(sub_word) for sub_word in self.tokenizer.tokenize(token)]
                for token in token_text]

        flat_sub_word_ids = [sub_word_id for token in sub_word_ids for sub_word_id in token]

        window_length = self._max_pieces - len(self._start_sub_word_ids) - len(self._end_sub_word_ids)

        offsets = []
        offset = len(self._start_sub_word_ids) if self._use_starting_offsets else len(self._start_sub_word_ids) - 1
        sub_word_accumulated = 0
        for token in sub_word_ids:
            next_offset = 1 if self._use_starting_offsets else 0
            if self._truncate_long_sequences and offset + len(token) - 1 >= window_length + next_offset:
                break

            if self._use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            else:
                offset += len(token)
                offsets.append(offset)

            sub_word_accumulated += len(token)

        if len(flat_sub_word_ids) <= window_length:
            sub_word_windows = [self._add_start_and_end(flat_sub_word_ids)]
        elif self._truncate_long_sequences:
            sub_word_windows = [self._add_start_and_end(flat_sub_word_ids[:sub_word_accumulated])]
        else:
            sub_word_windows = [self._add_start_and_end(flat_sub_word_ids)]

        sub_word_ids = [sub_word_id for sequence in sub_word_windows for sub_word_id in sequence]
        sub_word_mask = [1 for _ in sub_word_ids]
        mask = [1 for _ in offsets]
        return {
            index_name: sub_word_ids,
            f"{index_name}-offsets": offsets,
            f"{index_name}-mask": sub_word_mask,
            "mask": mask
        }

    def _add_start_and_end(self, sub_word_ids: List[int]) -> List[int]:
        return self._start_sub_word_ids + sub_word_ids + self._end_sub_word_ids

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val,
                                                             desired_num_tokens[key],
                                                             default_value=lambda: self._padding_value
                                                             if key == self._sub_word_indexer_name
                                                             else self._other_padding_value))
                for key, val in tokens.items()
                }
