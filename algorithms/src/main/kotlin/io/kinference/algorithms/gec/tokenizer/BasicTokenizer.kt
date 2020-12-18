package io.kinference.algorithms.gec.tokenizer

import io.kinference.algorithms.gec.tokenizer.utils.isControl
import io.kinference.algorithms.gec.tokenizer.utils.isPunctuation
import io.kinference.algorithms.gec.tokenizer.utils.isWhitespace
import io.kinference.algorithms.gec.tokenizer.utils.whitespaceTokenize
import java.lang.StringBuilder
import java.text.Normalizer
import java.util.ArrayList


/**
 * BasicTokenizer provides simple tokenization based on punctuation and accents
 * @param doLowerCase - boolean value for cased and uncased variants
 * @param stripAccents - boolean value for stripping accents
 */
class BasicTokenizer(private val doLowerCase: Boolean, private val stripAccents: Boolean) {

    fun tokenize(text: String): List<String> {
        val mText = cleanText(text)
        val origTokens = whitespaceTokenize(mText)
        val splitTokens = ArrayList<String>()

        for (token in origTokens) {
            var mToken = ""
            if (doLowerCase) {
                mToken = token.toLowerCase()
                if (stripAccents) {
                    mToken = runStripAccents(mToken)
                }
            } else if (stripAccents) {
                mToken = runStripAccents(token)
            }
            splitTokens += runSplitOnPunc(mToken)
        }
        return whitespaceTokenize(splitTokens.joinToString(" "))
    }

    private fun cleanText(text: String): String {
        val output = mutableListOf<String>()
        for (char in text) {
            val cp = char.toInt()
            if (cp == 0 || cp == 0xFFFD || isControl(char)) {
                continue
            }
            if (isWhitespace(char)) {
                output.add(" ")
            } else {
                output.add(char.toString())
            }
        }
        return output.joinToString("")
    }

    private fun runStripAccents(text: String): String {
        val mText = Normalizer.normalize(text, Normalizer.Form.NFD)
        val output = ArrayList<Char>()
        for (char in mText) {
            if (char.category == CharCategory.NON_SPACING_MARK) {
                continue
            }
            output.add(char)
        }
        return output.joinToString("")
    }

    private fun runSplitOnPunc(text: String): List<String> {
        val chars = text.toCharArray()
        var startNewWord = true
        val output = ArrayList<StringBuilder>()
        for (char in chars) {
            if (isPunctuation(char)) {
                output.add(StringBuilder().append(char.toString()))
                startNewWord = true
            } else {
                if (startNewWord) {
                    output.add(StringBuilder())
                }
                startNewWord = false
                output.last().append(char.toString())
            }
        }
        return output.map { it.toString() }
    }
}
