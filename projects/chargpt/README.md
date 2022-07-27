# chargpt

chargpt trains a character-level language model.

We support three settings: 1 convenience setting and 2 "benchmark" settings that have acedemic literature results:

- a user specified `input.txt` file that we train an LM on (e.g. get tiny-shakespear (1.1MB of data) [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))
- TODO [text8](http://mattmahoney.net/dc/textdata.html): also derived from Wikipedia text but all XML is removed and is lowercased to only 26 characters of
- TODO [enwik8](http://prize.hutter1.net) benchmark ("Hutter Prize"), first 100M bytes of a Wikipedia XML dump, with 205 unique tokensEnglish plus spaces
