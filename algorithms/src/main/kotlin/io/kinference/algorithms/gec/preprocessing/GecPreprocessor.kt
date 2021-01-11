package io.kinference.algorithms.gec.preprocessing

import io.kinference.algorithms.gec.utils.SentenceCorrections
import io.kinference.algorithms.gec.utils.Token
import io.kinference.algorithms.gec.utils.TokenRange
import io.kinference.algorithms.gec.utils.calculateTokensBordersAndWithSpaces

abstract class GecPreprocessor {
    abstract fun preprocess(sentId: Int, sentence: String): SentenceCorrections
}

class GecCorrectionPreprocessor(val textProcessor: TransformersTextprocessor,
                                val useStartToken: Boolean, val useForEval: Boolean) : GecPreprocessor() {

    override fun preprocess(sentId: Int, sentence: String): SentenceCorrections {
        val tokenizedSentence = evalTokenize(sentence)
        val tokensRanges: List<TokenRange> = calculateTokensBordersAndWithSpaces(text = sentence, tokens = tokenizedSentence, textWithSpace = false)

        val tokens = ArrayList<Token>()
        if (useStartToken) {
            tokens.add(Token(text = "\$START", encodedData = textProcessor.encodeAsIds("\$START"),
                tokenRange = TokenRange(start = 0, end = 0, withSpace = false), isFirst = false, isUsed = true))
        }
        for (idx in tokenizedSentence.indices) {
            val token = tokenizedSentence[idx]
            val tokenRange = tokensRanges[idx]
            tokens.add(Token(text = token,
                encodedData = textProcessor.encodeAsIds(token),
                tokenRange = tokenRange, isFirst = idx == 0, isUsed = isValidText(token)))
        }

        return SentenceCorrections(sentId, sentence, tokens)
    }

    private fun evalTokenize(sentence: String): List<String> {
        return sentence.split(" ")
    }

    private fun isValidText(text: String): Boolean {
        return true
    }
}
