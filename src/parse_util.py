def filter_by_len(parser, encoded, max_len):
    """Find indices of encoded sentences that are too long"""
    if parser.pretrained_model is not None:
        indexes_lens = [(index, len(one["input_ids"]))
                        for (index, one) in enumerate(encoded)]
    else:
        indexes_lens = [(index, len(one["valid_token_mask"]))
                        for (index, one) in enumerate(encoded)]
    indexes_out = [index for (index, len_) in indexes_lens
                   if len_ > max_len]
    return indexes_out


def make_dummy_parses(parser, examples, indexes_out):
    """Make dummy parses for sentences not to be parsed

    Parse is (TOP (FLAT (XX word) (XX word) ...))
    """
    examples_out = [example for (index, example) in enumerate(examples)
                    if index in indexes_out]
    trees_out = []
    for example in examples_out:
        leaves = [(word, 'XX') for (word, tag) in example.pos()]
        tree = parser.decoder.make_dummy_parse(leaves)
        trees_out.append(tree)
    return trees_out

def reinsert_dummy_parses(indexes_out, trees_out, test_predicted):
    """Put dummy parses back in results at the right locations"""
    index2tree = dict(zip(indexes_out, trees_out))
    predicted_iter = iter(test_predicted)
    new_lst = [index2tree[index] if index in index2tree else next(predicted_iter)
               for index in range(len(indexes_out) + len(test_predicted))]
    return new_lst

