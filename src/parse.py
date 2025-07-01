"""Parse a single file

Three main aspects to this:
(1) Reads in the sentences in three different formats
(2) Duplicates parser handling of sentence with ParsingExample
(3) Checks if sentences are too long to parse, and if so substitutes with
dummy flat parse
"""
import sys
import argparse
import pathlib
import itertools
import torch
from benepar import parse_chart
from benepar.parse_base import BaseInputExample
from benepar.ptb_unescape import ptb_unescape, guess_space_after
from parse_util import filter_by_len
from parse_util import make_dummy_parses
from parse_util import reinsert_dummy_parses

class SentenceWrapper(BaseInputExample):
    """Class to take place of ParsingExample for parser

    words are what what the parser uses
    leaves are used to construct the tree
    """
    def __init__(self, words, leaves, space_after):
        # words are form for the parser
        # leaves are as it should be the tree

        self.words = words
        self._leaves = leaves
        self.space_after = space_after

    @property
    def tree(self):
        return None

    def leaves(self):
        return self._leaves

    def pos(self):
        return [(leaf, "UNK") for leaf in self._leaves]

def make_sentence_wrapper(text, text_processing):
    """Duplicate parser text processing for input text

    Since these are not trees being read in, they won't have -LRB- and
    -RRB-.  So do that substitution and then they look
    like the leaves of a tree read in.  The processing then follows
    the (modified) treebanks
    """
    words_tmp = text.split()
    leaves = [word.replace('(', '-LRB-').replace(')', '-RRB-')
              for word in words_tmp]
    words = ptb_unescape(leaves)

    if text_processing == 'default':
        sp_after = guess_space_after(leaves)
        sp_after[-1] = False
    #elif text_processing == 'ftd':
    #    sp_after = guess_space_after_ftd(leaves)
    else:
        sp_after = [True for _ in words]
        sp_after[-1] = False
    return SentenceWrapper(words, leaves, sp_after)


def read_file_conll(fname, text_processing, word_col):
    """Read a file in our almost conll format"""
    def _is_divider(line):
        return line.strip() == ""
    sents = []
    with open(fname, 'r', encoding='utf-8') as fin:
        for is_divider, lines in itertools.groupby(fin, _is_divider):
            if not is_divider:
                lines2 = [line.rstrip('\n') for line in lines]
                assert lines2[0].startswith('SENT'), 'something weird reading in pos'
                lines2b = [line.split('\t') for line in lines2[1:]]
                #text = ' '.join([word for [_, word, _] in lines2b])
                text = ' '.join([line[word_col] for line in lines2b])
                sents.append(make_sentence_wrapper(text, text_processing))
    return sents

def read_file_text(fname, text_processing):
    """Read a file with one text to a line"""
    with open(fname, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.rstrip('\n') for line in lines]
    sents = [make_sentence_wrapper(line, text_processing) for line in lines]
    return sents


def read_file_eebo(fname, text_processing):
    """Read a file with one text to a line"""
    with open(fname, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]
    lines = [line[4] for line in lines]

    sents = [make_sentence_wrapper(line, text_processing) for line in lines]
    return sents


def do_one_file(parser, sents, max_word_len, max_subword_len, subbatch_max_tokens):
    """Get parses for one input list of sents"""
    if max_word_len > 0:
        print(f'filtering for len {max_word_len}')
        print(f'length before filtering {len(sents)}')
        sents = [sent for sent in sents if len(sent.words) <= max_word_len]
        print(f'length after filtering {len(sents)}')
        filtered_sents = sents
        filtered_encoded = None
        indexes_out = []
    else:
        encoded = parser.encode_examples(sents)
        print(f"Checking length of test sentences for max_subword_len {max_subword_len} "
              f"# examples={len(sents)}")
        indexes_out = filter_by_len(parser, encoded, max_subword_len)
        print(f'#sentences removing for length: {len(indexes_out)}')

        if not indexes_out:
            filtered_sents = sents
            filtered_encoded = encoded
        else:
            # output some info about them
            lst = [f'{index}({len(sents[index].leaves())})'
                   for index in indexes_out]
            print(' '.join(lst))
            # make the dummyparses to put in later
            dummy_parses = make_dummy_parses(
                parser, sents, indexes_out)
            # and filter them out from both sents and encoded
            # (which must be the same length)
            filtered_sents = [example for (index, example) in enumerate(sents)
                                   if index not in indexes_out]
            filtered_encoded = [one for (index, one) in enumerate(encoded)
                                  if index not in indexes_out]

    # modified this to allow encoded list to be sent in, so don't need
    # to do it again
    filtered_predicted = parser.parse(
        filtered_sents,
        encoded=filtered_encoded,
        subbatch_max_tokens=subbatch_max_tokens,
      )

    if indexes_out:
        # and now put the dummy parses back in
        test_predicted = reinsert_dummy_parses(
            indexes_out, dummy_parses, filtered_predicted)
    else:
        test_predicted = filtered_predicted

    return test_predicted

def run_parse(args):
    """Parse sentence from a file or list of files"""
    model_path = args.model_path[0]
    print("Loading model from {}...".format(model_path))
    parser = parse_chart.ChartParser.from_trained(model_path)
    if args.parallelize:
        parser.parallelize()
    elif torch.cuda.is_available():
        parser.cuda()

    in_fname = args.test_path
    out_fname = args.output_path
    print(f'in  {in_fname}\nout {out_fname}')

    if args.in_format == 'text':
        sents = read_file_text(in_fname, args.text_processing)
    elif args.in_format == 'conll':
        sents = read_file_conll(in_fname, args.text_processing, args.word_col)
    elif args.in_format == 'eebo':
        sents = read_file_eebo(in_fname, args.text_processing)
    else:
        print(f'unknown in_format {args.in_format}')
        sys.exit(-1)

    if args.max_word_len > 0:
        max_subword_len = 0
    else:
        max_subword_len = parser.determine_max_len()

    print(f'max_word_len={args.max_word_len} '
          f'max_subword_len={max_subword_len}')

    output_trees = do_one_file(
        parser, sents, args.max_word_len, max_subword_len, args.subbatch_max_tokens)

    with open(out_fname, "w", encoding='utf-8') as outfile:
        for tree in output_trees:
            outfile.write("{}\n".format(tree.pformat(margin=1e100)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", nargs="+", required=True)
    parser.add_argument("--test-path", type=pathlib.Path)
    parser.add_argument("--output-path", type=pathlib.Path)

    parser.add_argument("--text-processing", default="default")
    parser.add_argument("--in-format", type=str, default='text')
    parser.add_argument('--word-col', '-w',  type=int, default=1,
                        help='column for conll file')    
    parser.add_argument("--max-word-len", type=int, default=0)

    parser.add_argument("--subbatch-max-tokens", type=int, default=500)
    parser.add_argument("--parallelize", action="store_true")

    args = parser.parse_args()
    run_parse(args)

if __name__ == "__main__":
    main()
