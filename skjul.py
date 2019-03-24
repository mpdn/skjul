#!/usr/bin/env python3
"""Text-based steganography.

Usage:
  skjul.py process [<in>] [<pairs>] [--neighbors=<k>] [--lines=<n>]
  skjul.py encode <secret> [<pairs>] [--key=<k>] [--noise=<x>]
  skjul.py decode [<pairs>] [--key=<k>] [--noise=<x>]
  skjul.py --version

Commands:
  process             Build a pairs list from a fastText vector file.
  encode              Encode a secret message. Carrier is read from standard
                      input and output is written to standard output.
  decode              Decode a secret messasge from standard input.

All commands accept a path to a pair-list file. If this is not supplied, then
'skjul.csv' in the current working directory is used instead.

Options:
  -h --help           Show this screen.
  --version           Show version.
  -n --lines=<n>      Number of lines to read from file [default: all].
  -k --neighbors=<k>  Number of neighbors to find for each word [default: 10].
  -k --key=<key>      Key to encode/decode message with [default: 0].
  -x --noise=<x>      Noise fraction when selecting words [default: 0.025].
"""

import re
import csv
from enum import Enum
import numpy as np
from sklearn.neighbors import NearestNeighbors
from docopt import docopt
from schema import Schema, And, Or, Use, Regex
import sys
import itertools


def _pairing(x, k=10, metric='cosine', stable=False):
    """
    Pairs points such that points that are closer wrt. the given metric are
    more likely to be paired together.

    Args:
        x (array): A n by d matrix representing n vectors of length d.
        k (int): Number of neighbors to consider when pairing.
        metric (str): Metric to use for nearest neighbor.
        stable (bool): Whether to ensure exact output for the given input.

    See sklearn.neighbors.NearestNeighbors for all possible metrics.

    Returns:
        low: Lower indices of each pair
        hi: Higher indices of each pair
        dist: Distance between each point
    """

    x = np.asarray(x)
    n = x.shape[0]

    edge_dist, edge_tgt = NearestNeighbors(n_neighbors=k, metric=metric) \
        .fit(x).kneighbors()

    sorting = np.argsort(edge_dist, axis=None,
                         kind='stable' if stable else None)

    edge_src = np.unravel_index(sorting, edge_dist.shape)[0]
    edge_tgt = np.ravel(edge_tgt)[sorting]
    edge_dist = np.ravel(edge_dist)[sorting]

    pairing = np.full([n], -1, dtype=np.int32)
    pairing_dist = np.zeros([n], dtype=edge_dist.dtype)

    for src, tgt, dist in np.nditer((edge_src, edge_tgt, edge_dist)):
        if pairing[src] == -1 and pairing[tgt] == -1:
            pairing[src] = tgt
            pairing[tgt] = src
            pairing_dist[src] = pairing_dist[tgt] = dist

    paired_indices = np.where(pairing != -1)[0]
    lo = paired_indices[pairing[paired_indices] > paired_indices]
    hi = pairing[lo]

    return lo, hi, pairing_dist[lo]


def _gamma_encode(num):
    """
    Encodes a positive number using Elias gamma coding.

    Args:
        num (int): An integer to encode. Must be positive.

    Returns:
        list: A list of booleans representing bits of the encoded number.
    """
    code = [False] * num.bit_length()

    for i in range(num.bit_length() - 1, -1, -1):
        code.append((num >> i) & 1 != 0)

    return code


def _gamma_decode(bits):
    """
    Decodes an Elias gamma encoded number.

    Args:
        bits (iterable): An iterable of bits to decode.

    Returns:
        int: The decoded gamma integer.
    """

    reading = False
    length = 0
    num = 0

    for bit in bits:
        if not reading:
            if bit:
                reading = True
            else:
                length += 1

        if reading:
            num = num << 1
            num |= 1 if bit else 0
            length -= 1

            if length == 0:
                return num


class _Caps(Enum):
    UPPER = 1
    TITLE = 2
    LOWER = 3

    @staticmethod
    def from_word(word):
        if word.istitle():
            return _Caps.TITLE
        elif word.isupper():
            return _Caps.UPPER
        else:
            return _Caps.LOWER

    def apply(self, word):
        if self == _Caps.UPPER:
            return word.upper()
        elif self == _Caps.TITLE:
            return word.title()
        else:
            return word.lower()


class Steganographer:
    """
    A class for hiding secret messages in ordinary text.
    """

    TOKEN_REGEX = re.compile(r'\w+')

    @staticmethod
    def from_embeddings(words, embeddings, k=5, metric='cosine'):
        """
        Creates a new steganographer from a list of words and corresponding
        embeddings.

        Args:
            words (list): A list of words of length n.
            embeddings (array): A n by d matrix of word embeddings.
            k (int): The number of neighbors to consider when pairing words.
            metric (str): The metric to use when pairing words.

        Returns:
            Steganographer: A new steganographer.
        """

        words = np.asarray(words)
        embeddings = np.asarray(embeddings)
        lower_words = {}
        valid = np.zeros([words.size], np.bool)

        # Filter non-token words and lowercase all words. In case of a
        # collision, prefer embeddings from lowercase words. We assume that
        # these are more common and therefore more representative.

        for i, word in enumerate(words):
            lower = word.lower()

            valid[i] = (Steganographer.TOKEN_REGEX.fullmatch(word) is not None
                        and (lower not in lower_words or not word.islower()))

            if valid[i]:
                old = lower_words.get(lower)

                if old is not None:
                    valid[old] = False

                lower_words[lower] = i

        words = np.char.lower(words[valid])
        embeddings = embeddings[valid]

        left, right, dist = _pairing(embeddings, k, metric=metric, stable=True)
        return Steganographer(zip(words[left], words[right], dist))

    @staticmethod
    def load(file):
        """Loads a steganographer from a file-like object"""

        return Steganographer((a, b, float(dist))
                              for [a, b, dist]
                              in csv.reader(file))

    def __init__(self, pairs):
        self.map = {a: (b, dist, value)
                    for left, right, dist in pairs
                    for a, b, dist, value in [(left, right, dist, True),
                                              (right, left, dist, False)]}

    def save(self, file):
        """Saves the steganographer to a file-like object"""

        pairs = ([a, b, dist]
                 for (a, (b, dist, value))
                 in self.map.items()
                 if value)

        csv.writer(file).writerows(sorted(pairs, key=lambda x: x[2]))

    def _tokenize(self, string):
        intertokens = []
        tokens = []
        caps = []

        last_end = 0

        for match in Steganographer.TOKEN_REGEX.finditer(string):
            if match.group().lower() in self.map:
                intertokens.append(string[last_end:match.start()])
                tokens.append(match.group().lower())
                caps.append(_Caps.from_word(match.group()))
                last_end = match.end()

        intertokens.append(string[last_end:])

        return tokens, caps, intertokens

    def encode(self, carrier, secret, key=0, noise=0):
        """
        Encodes a secret message into a carrier message.

        Args:
            carrier (str): A string to embed a secret message into.
            secret (list): A list of booleans to embed.
            key (int): A key to encode the message with.
            noise (float): The amount of noise to add.

        Returns:
            str: A message with the given secret embedded within it.
        """
        tokens, caps, intertokens = self._tokenize(carrier)

        rng = np.random.RandomState(key)
        token_noise = rng.rand(len(tokens)) * noise

        secret = _gamma_encode(len(secret)) + secret

        if len(secret) > len(tokens):
            raise ValueError('Insufficient tokens for secret')

        dist = np.array([self.map[token][1] for token in tokens]) + token_noise

        for index, bit in zip(np.argsort(dist), secret):
            paired, _, value = self.map[tokens[index]]
            if (value != bool(bit)) != bool(rng.randint(1)):
                tokens[index] = paired

        result = []
        for token, cap, intertoken in zip(tokens, caps, intertokens):
            result.append(intertoken)
            result.append(cap.apply(token))

        result.append(intertokens[-1])

        return ''.join(result)

    def decode(self, message, key=0, noise=0):
        """
        Extracts a secret message from a string.

        Args:
            message (str): A string from which to extract a secret message.
            key (int): The key the message was encoded with.
            noise (float): The amount of noise the message was encoded with.

        Returns:
            list: The decoded secret list of booleans.
        """
        tokens = self._tokenize(message)[0]

        rng = np.random.RandomState(key)
        token_noise = rng.rand(len(tokens)) * noise

        dist = np.array([self.map[token][1] for token in tokens]) + token_noise

        bits = (self.map[tokens[index]][2] != bool(rng.randint(1))
                for index in np.argsort(dist))

        secret_len = _gamma_decode(bits)

        return list(itertools.islice(bits, secret_len))


def _read_fast(file, nrows=None):
    """
    Reads a facebook fastText formatted vector file into a list of words and a
    2d numpy array of corresponding embeddings.
    """
    [n, d] = [int(s) for s in file.readline().split(' ')]

    if nrows is not None:
        n = min(n, nrows)

    embeddings = np.zeros([n, d], np.float32)
    words = []

    for i, line in enumerate(file):
        if i >= n:
            break

        row = line.split(' ')
        words.append(row[0])
        embeddings[i, :] = [float(x) for x in row[1:d + 1]]

    return words, embeddings


def main():
    raw_args = docopt(__doc__, version='skjul 0.1')

    schema = Schema({
        'process': bool,
        'encode': bool,
        'decode': bool,
        '<in>': Or(None, Use(open)),
        '<pairs>': Use(lambda x: x or 'skjul.csv'),
        '<secret>': Or(None, And(Regex(r'^[01]*$'), Use(
            lambda s: [c == '1' for c in s]))),
        '--version': bool,
        '--lines': Or(And('all', Use(lambda x: None)), Use(int)),
        '--neighbors': Use(int),
        '--key': Use(int),
        '--noise': Use(float),
    })

    args = schema.validate(raw_args)

    if args['process']:
        words, embeddings = _read_fast(args['<in>'] or sys.stdin,
                                       nrows=args['--lines'])

        st = Steganographer.from_embeddings(words, embeddings,
                                            k=args['--neighbors'])

        with open(args['<pairs>'], 'w') as pairs:
            st.save(pairs)

    elif args['encode']:
        with open(args['<pairs>']) as pairs:
            st = Steganographer.load(pairs)

        sys.stdout.write(st.encode(sys.stdin.read(), args['<secret>'],
                                   args['--key'], args['--noise']))
    elif args['decode']:
        with open(args['<pairs>']) as pairs:
            st = Steganographer.load(pairs)
        decoded = st.decode(sys.stdin.read(), args['--key'], args['--noise'])
        sys.stdout.write(''.join('1' if bit else '0' for bit in decoded ))


if __name__ == "__main__":
    main()
