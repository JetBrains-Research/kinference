package io.kinference.algorithms.gec.tokenizer.word

import io.kinference.algorithms.gec.tokenizer.Tokenizer
import io.kinference.algorithms.gec.tokenizer.utils.*
import java.text.Normalizer


/**
 * BasicTokenizer is a word tokenizer that provides simple tokenization based on punctuation and accents
 * @param toLowerCase should all the output tokens be lower-cased
 * @param shouldNormalizeAccentLetters should stripping of accents (like letters with umlaut) be performed
 */
class BasicWordTokenizer(private val toLowerCase: Boolean, private val shouldNormalizeAccentLetters: Boolean) : Tokenizer {
    override fun tokenize(text: String): List<String> {
        val mText = cleanText(text)
        val origTokens = mText.tokenizeByWhitespace()
        val splitTokens = ArrayList<String>()

        for (token in origTokens) {
            var mToken = token
            if (toLowerCase) {
                mToken = mToken.toLowerCase()
                if (shouldNormalizeAccentLetters) {
                    mToken = normalizeAccents(mToken)
                }
            } else if (shouldNormalizeAccentLetters) {
                mToken = normalizeAccents(mToken)
            }
            splitTokens += splitOnPunctuation(mToken)
        }
        return splitTokens.flatMap { it.tokenizeByWhitespace() }
    }

    private fun cleanText(text: String): String = buildString {
        for (char in text) {
            val cp = char.toInt()
            if (cp == 0 || cp == 0xFFFD || CharUtils.isControl(char)) {
                continue
            }
            if (char.isWhitespace()) {
                append(" ")
            } else {
                append(char)
            }
        }
    }

    private fun normalizeAccents(text: String): String = buildString {
        val mText = Normalizer.normalize(text, Normalizer.Form.NFD)
        for (char in mText) {
            if (char.category == CharCategory.NON_SPACING_MARK) {
                continue
            }
            append(char)
        }
    }

    private fun splitOnPunctuation(text: String): List<String> {
        var startNewWord = true
        val output = ArrayList<StringBuilder>()
        for (char in text) {
            if (CharUtils.isPunctuation(char)) {
                output.add(StringBuilder().append(char))
                startNewWord = true
            } else {
                if (startNewWord) {
                    output.add(StringBuilder())
                }
                startNewWord = false
                output.last().append(char)
            }
        }
        return output.map { it.toString() }
    }
}
