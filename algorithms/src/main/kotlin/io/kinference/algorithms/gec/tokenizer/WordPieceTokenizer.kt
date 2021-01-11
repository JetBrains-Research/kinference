package io.kinference.algorithms.gec.tokenizer

import io.kinference.algorithms.gec.tokenizer.utils.whitespaceTokenize

/**
 * Implementation of transformers WordPieceTokenizer which based on vocabulary
 * @param vocab - vocabulary map <Token, TokenIndex>
 * @param unkToken - token for unknown words
 * @param maxInputCharsPerWord - maximum number of characters in word
 */
class WordPieceTokenizer(val vocab: Map<String, Int>, val unkToken: String, val maxInputCharsPerWord: Int = 100) {
    fun tokenize(text: String): List<String> {
        val out = ArrayList<String>()

        for (token in whitespaceTokenize(text)) {
            val chars = token.toCharArray()
            if (chars.size > maxInputCharsPerWord) {
                out.add(unkToken)
                continue
            }
            var isBad = false
            var start = 0
            val subTokens = ArrayList<String>()
            while (start < chars.size) {
                var end = chars.size
                var curSubstr: String? = null
                while (start < end) {
                    var substr = chars.slice(start until end).joinToString("")
                    if (start > 0) {
                        substr = "##$substr"
                    }
                    if (vocab.containsKey(substr)) {
                        curSubstr = substr
                        break
                    }
                    end -= 1
                }
                if (curSubstr == null) {
                    isBad = true
                    break
                }
                subTokens.add(curSubstr)
                start = end
            }
            if (isBad) {
                out.add(unkToken)
            } else {
                for (item in subTokens) {
                    out.add(item)
                }
            }
        }
        return out
    }
}
