# Skjul - Text-based steganography

*Steganography* is the practice of inconspicuously hiding data (a secret) within
some other data (a carrier). Often this is within images, where the lower bits
can be used to store a secret message. While having few real uses, steganography
can be a fun exercise in information theory.

Skjul (danish for *hide* as in *to hide*), is a text-based steganography
implementation. Given a carrier message, Skjul can encode a secret bitstring
into it by slightly changing words - hopefully so little as to be imperceptible
to an uninitiated reader.

## Example

    $ cat example.txt
    ‘The Babel fish,’ said The Hitchhiker’s Guide to the Galaxy quietly, ‘is
    small, yellow, and leech-like, and probably the oddest thing in the
    Universe. It feeds on brainwave energy received not from its own carrier but
    from those around it. It absorbs all unconscious mental frequencies from
    this brainwave energy to nourish itself with. It then excretes into the mind
    of its carrier a telepathic matrix formed by combining the conscious thought
    frequencies with nerve signals picked up from the speech centres of the
    brain which has supplied them. The practical upshot of all this is that if
    you stick a Babel fish in your ear you can instantly understand anything
    said to you in any form of language.

    $ cat example.txt | ./skjul.py encode '101010' | tee 'encoded.txt'
    ‘The Babel fish,’ said The Hitchhiker’s Guide to the Galaxy quietly, ‘is
    small, yellow, and leech-like, and possibly the oddest thing in the
    Universe. It feeds on brainwave energy recieved not to its own carrier but
    to those around it. It absorbs all unconscious mental frequencies from this
    brainwave energy to nourish itself with. It then excretes into the mind of
    its carrier a telepathic matrix formed by combining the conscious thought
    frequencies with nerve signals picked up from the speech centres of the
    brain which has supplied them. The practical upshot of all that is that if
    you stick a Babel fish in your ear you can instantly comprehend anything
    said from you in any form of language.

    $ wdiff example.txt encoded.txt
    ‘The Babel fish,’ said The Hitchhiker’s Guide to the Galaxy quietly, ‘is
    small, yellow, and leech-like, and [-probably-] {+possibly+} the oddest
    thing in the Universe. It feeds on brainwave energy [-received-]
    {+recieved+} not [-from-] {+to+} its own carrier but [-from-] {+to+} those
    around it. It absorbs all unconscious mental frequencies from this brainwave
    energy to nourish itself with. It then excretes into the mind of its carrier
    a telepathic matrix formed by combining the conscious thought frequencies
    with nerve signals picked up from the speech centres of the brain which has
    supplied them. The practical upshot of all [-this-] {+that+} is that if you
    stick a Babel fish in your ear you can instantly [-understand-]
    {+comprehend+} anything said [-to-] {+from+} you in any form of language.

    $ cat encoded.txt | ./skjul decode
    101010

The secrets can only be bitstrings.

## How it works

### Word pairs

Given a carrier string and a secret bitstring, the basic idea is assign to each
word in the carrier string a *paired word*. The secret message can then be
encoded in our choice of word. To not have a noticeable difference, the paired
word should be able to "work" in the same context as the original word, i.e. we
wish to select words that are likely to share the same neighboring words.

Word-vector models is a common way to model these *distributional properties* of
words. In such a model, each word has a vector embedding of e.g. 300 dimensions.
These embeddings are built such that words that tend to have similar contexts
also tend to have similar embeddings.

Using a word vector model, we pair each embedding with a neighbor using cosine
distance as the metric. Note that these pairings must be exclusive, i.e.
`[(a,b), (a,c)]` is not valid because `a` participates in both pairs. Instead,
we find the k-nearest neighbors and then greedily pair words based on the
distance to the closest non-paired k-neighbor. This means that words are not
always paired with their closest neighbor and some words are not paired at all.

This repository includes a precomputed pair list based on
[Facebook's fasttext vectors](https://fasttext.cc/docs/en/crawl-vectors.html).

For example, the string has "this is a test" has 3 words in the pair list:

| 1    | 0     | Distance   |
|------|-------|------------|
| this | that  | 0.17533547 |
| was  | is    | 0.28453428 |
| test | tests | 0.20037645 |

To encode a *k*-bit message, we simply pick the *k* tokens with lowest distance
to their paired word and swap or not depending on the corresponding bit in the
secret. Eg. to encode a single 1-bit, we change "this is a test" to "that is a
test".

### Variable length coding

The method as outlined above requires the person decoding to know the length of
the secret. This makes it somewhat unpractical and it would be better to encode
the length as part of the message itself. To do this, we need a prefix-free
encoding scheme, as we do not know the amount of bits for the length beforehand.

For this, we use
[Elias gamma coding](https://en.wikipedia.org/wiki/Elias_gamma_coding). In gamma
coding, we first encode the length of the integer in unary zero bits followed by
the integer iteself.

A downside of this is that it increases the length of the secret, especially for
small secrets. This is due to how the number of bits in the length itself is
comparatively more significant to the number of bits in a short secret than in a
long secret.

### A pinch of noise

Lastly, we add XOR encryption to the secret using a pseudorandom number
generator (PRNG). This breaks any predictable patterns that might be in the
secret. For example, a secret of only zeros and a carrier that contains the same
pair often would always pick the same word. This also makes it possible to
specify a key by using the key as seed for the PRNG.

We also add an small, optional amount of noise to each word pair distance. This
make the pairs chosen more varied, such that it is not always the minimum word
that is chosen.