import random
from copy import deepcopy
from benepar import ptb_unescape
from treebanks import Treebank, ParsingExample

CHANGE0_PCT = 0.85
CHANGE1_PCT = 0.10
CHANGE2_PCT = 0.05
CHANGE_PCT_LST = [
    CHANGE0_PCT,
    CHANGE0_PCT + CHANGE1_PCT,
    CHANGE0_PCT + CHANGE1_PCT + CHANGE2_PCT]


PUNC_TAGS = [
    ',',
    '.',
    'OPAREN',
    'CPAREN',
    '"',
    "'",
    ]



def transform(treebank):

    """
    Each example has a nltk tree, with words and sp_after derived from that.
    """
    new_examples = [transform_example(example)
                    for example in treebank]
    return Treebank(new_examples)

def transform_example(example):
    # list of (word, pos)
    word_pos_lst = example.pos()

    tree2 = deepcopy(example.tree)    
    leaf_posits = tree2.treepositions('leaves')
    assert len(word_pos_lst) == len(leaf_posits), \
        f'something weird {word_pos_lst} {len(leaf_posits)}'
    
    for num in range(len(word_pos_lst)):
        (_, pos) = word_pos_lst[num]
        if pos not in PUNC_TAGS:
            leaf = tree2[leaf_posits[num][:-1]]
            word = leaf[0]
            if word != example.words[num]:
                print(f'unexpected inconsistency {pos} {word} {example.words[num]}')
            new_word = mod_word(word)
            if new_word is not None:
                leaf[0] = new_word
    words = ptb_unescape.ptb_unescape(tree2.leaves())
    space_after = deepcopy(example.space_after)
    return ParsingExample(tree=tree2, words=words, space_after=space_after)

def mod_word(word):
    num = random.choices([0,1,2], cum_weights=CHANGE_PCT_LST, k=1)
    num = num[0]
    if num == 2 and len(word) == 1:
        num = 1
    if num == 0:
        return None
    if num == 1:
        index = random.randrange(0,len(word))
        word = word[:index] + u'\u2022' + word[index+1:]
        return word
    index = random.randrange(0,len(word)-1)
    word = word[:index] + u'\u2022' + u'\u2022' + word[index+2:]
    return word
