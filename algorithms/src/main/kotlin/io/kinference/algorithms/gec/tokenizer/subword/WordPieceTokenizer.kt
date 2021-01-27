package io.kinference.algorithms.gec.tokenizer.subword

import io.kinference.algorithms.gec.tokenizer.Tokenizer
import io.kinference.algorithms.gec.tokenizer.utils.tokenizeByWhitespace

/**
 * WordPieceTokenizer is a sub-word tokenizer, mostly it is used in transformers and is based on vocabulary
 * @param vocabulary vocabulary map <Token, TokenIndex>
 * @param unknownToken token for unknown words
 * @param maxInputCharsPerWord  maximum number of characters in word
 */
class WordPieceTokenizer(private val vocabulary: Map<String, Int>, private val unknownToken: String, private val maxInputCharsPerWord: Int = 100): Tokenizer {
    override fun tokenize(text: String): List<String> {
        val out = ArrayList<String>()

        for (token in text.tokenizeByWhitespace()) {
            if (token.length > maxInputCharsPerWord) {
                out.add(unknownToken)
                continue
            }

            var isUnknown = false
            var start = 0
            val subTokens = ArrayList<String>()
            while (start < token.length) {
                var end = token.length
                var curSubstr: String? = null
                while (start < end) {
                    var substr = token.slice(start until end)
                    if (start > 0) {
                        substr = "##$substr"
                    }
                    if (vocabulary.containsKey(substr)) {
                        curSubstr = substr
                        break
                    }
                    end -= 1
                }
                if (curSubstr == null) {
                    isUnknown = true
                    break
                }
                subTokens.add(curSubstr)
                start = end
            }
            if (isUnknown) {
                out.add(unknownToken)
            } else {
                out.addAll(subTokens)
            }
        }
        return out
    }
}
