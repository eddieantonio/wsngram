#!/usr/bin/env python
# coding: utf-8

# Copyright 2014 Eddie Antonio Santos
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tokenizes Python. Includes whitespace in the token stream, as its actual text.

Model Building Strategies:
 - Interleaved whitespace -- Tokens and whitespace are interleaved.
 - Last-only -- the only whitespace appears as the nth element in the n-gram,
   for prediction.
 - 1-twee-n-gram -- one suffix token is kept

Testing:
 - ALL permutations!
 - Control
"""

import itertools
import token
import tokenize

from collections import namedtuple

# Strategies
STRATEGIES = {
    'order': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    'interleaved': (True,),
    'tweengrams': (0,),
    'use_text': (True, False),
    'smoothing': (False,),
    #'interleaved': (True, False),
    #'tweengrams': (0, 1),
    #'smoothing': (True, False),
}

TOKEN_FIELDS = 'type text start end line'

ALWAYS_CATEGORY = (token.NEWLINE, token.INDENT, token.DEDENT, tokenize.NL,
        token.ERRORTOKEN, token.ENDMARKER)

CATEGORY     = 0
TEXT         = 1
START        = 2
END          = 3
LOGICAL_LINE = 4

ROW = 0
COLUMN = 1


# Singleton 'Start' token. It's always the start of any sequence.
Start = namedtuple('Start', TOKEN_FIELDS)('START',
                                          '',
                                          (1, 1),
                                          (1, 1),
                                          1)
Whitespace = namedtuple('Whitespace', TOKEN_FIELDS)

def whitespace_tokenize(source_lines):
    lineiter = iter(source_lines)
    readline = lambda: next(lineiter)

    # Create a list of tokens WITHOUT
    tokens_sans_nl = (token for token in tokenize.generate_tokens(readline)
                      if token[0] != tokenize.NL)

    tokens_with_start = itertools.chain((Start,), tokens_sans_nl)

    def generate():
        # Generate the whitespace and the token. Since the very first token in
        # the sequnce is START, it gets silently thrown out.
        for prev_token, token in bigrams(tokens_with_start):
            yield gen_whitespace(prev_token, token, source_lines)
            yield token
    return list(generate())

def bigrams(sequence):
    """
    Bigrams for iterables.

    >>> list(bigrams([1,2,3]))
    [(1, 2), (2, 3)]

    """

    normal, peek = itertools.tee(sequence, 2)
    # Get rid of one token.
    next(peek)
    return itertools.izip(normal, peek)



def gen_whitespace(token, next_token, source_lines):
    # Start is end of token.
    srow, scol = token[END]
    start = (srow, scol + 1)
    # End is start of next token.
    erow, ecol = next_token[START]
    end = (erow, ecol - 1)

    # The tokens are on different lines.
    if start[COLUMN] == end[COLUMN]:
        line = source_lines[start[ROW] - 1]
        text = line[start[COLUMN]:end[COLUMN]]
    else:
        lines_between = erow - srow
        end_line = source_lines[erow - 1] if (erow - 1) < len(source_lines) else ''
        text = '\n' * lines_between + end_line[:ecol]

    return Whitespace('WHITESPACE', text, start, end, token[LOGICAL_LINE])

def use_text(token):
    if token[0] in ALWAYS_CATEGORY:
        return '<%s>' % (canonical_category_name(token),)
    return token[1]

def use_category(token):
    return (token.tok_name.get(token[0], token[0]),)

def canonical_category_name(token):
    "Returns the 'canonical' name of the token's category."
    return token.tok_name.get(token[1], token[1])

def interleave_ws(sequence, order):
    """
    n-grams interleaving whitespace tokens between normal tokens.

    >>> sequence = whitespace_tokenize(['if False:'])
    >>> map(type, next(iter(interleave_ws(sequence, 3))))
    [<class 'wsngram.Whitespace'>, <type 'tuple'>, <class 'wsngram.Whitespace'>]
    """

    # Skip every... thing.
    for i in xrange(0, len(sequence) - order, 2):
        ngram = sequence[i:i + order]
        assert type(ngram[-1]) is Whitespace
        assert len(ngram) == order
        yield ngram

def ws_last(sequence, order):
    """
    n-grams where the last token is a whitespace token.
    """
    for i in xrange(0, len(sequence) - order, 2):
        # The index of the last whitespace token.
        last_ws_pos = i + (order - 1) * 2
        # Skip every second token.
        ngram = sequence[i:last_ws_pos:2]
        assert not any(type(token) is Whitespace for token in ngram)

        ws_token = sequence[last_ws_pos]
        assert type(ws_token) is Whitespace
        ngram.append(ws_token)

        assert len(ngram) == order
        yield ngram

def tween(ngram, t=1):
    """
    >>> tween([1, 2, 3, 4, 5])
    [1, 2, 3, 5, 4]
    >>> tween([1, 2, 3, 4, 5], t=2)
    [1, 2, 4, 5, 3]
    """
    last = ngram[-(t + 1)]

    prefix = ngram[:-(t + 1)]
    suffix = ngram[-t:]
    return prefix + suffix + [last]


def simplify(token, store_text_func=use_category):
    """
    Simplifies the token.
    """

    if isinstance(token, Whitespace):
        return ('W', token.text)
    return ('T', store_text_func(token))

def make_ngrams(sequence, order=None, interleaved=None, tweengrams=None, use_text=None):
    """
    Creates n-grams.... very carefully!

    Assumes the input is already interleaved with whitespace tokens.
    """

    # Pad the start with START tokens.
    #start_padding = [Start] * (order - 1)
    #full_sequence = list(itertools.chain(start_padding, sequence))

    # The ngramming thingy...
    ngrammer = interleave_ws if interleaved else ws_last
    store_text = get_text if use_text else use_category

    for ngram in ngrammer(full_sequence, order):
        if tweengrams:
            ngram = tween(ngram, tweengrams)
        yield [simplify(token, store_text) for token in ngram]

