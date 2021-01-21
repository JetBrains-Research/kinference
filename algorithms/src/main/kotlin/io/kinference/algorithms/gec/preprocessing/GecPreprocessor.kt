package io.kinference.algorithms.gec.preprocessing

import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.tokenizer.utils.tokenizeByWhitespace
import io.kinference.algorithms.gec.utils.SentenceCorrections
import io.kinference.algorithms.gec.utils.Token
import io.kinference.algorithms.gec.utils.Token.TokenRange
import io.kinference.algorithms.gec.utils.calculateTokensBordersAndWithSpaces

/** Preprocessor is used to tokenize and pre-filter tokens and generation input for first iteration  */
interface GecPreprocessor {
    fun preprocess(sentId: Int, sentence: String): SentenceCorrections
}

/** Basic preprocessor implementation for inference only */
class GecCorrectionPreprocessor(
    val encoder: PreTrainedTextEncoder,
    val useStartToken: Boolean
) : GecPreprocessor {

    override fun preprocess(sentId: Int, sentence: String): SentenceCorrections {
        val tokenizedSentence = sentence.tokenizeByWhitespace()
        val tokensRanges: List<TokenRange> = calculateTokensBordersAndWithSpaces(text = sentence, tokens = tokenizedSentence, textWithSpace = false)

        val tokens = ArrayList<Token>()
        if (useStartToken) {
            tokens.add(
                Token(
                    text = "\$START", encodedData = encoder.encodeAsIds("\$START", false),
                    tokenRange = TokenRange(start = 0, end = 0, withSpace = false), isFirst = false, isUsed = true
                )
            )
        }
        for (idx in tokenizedSentence.indices) {
            val token = tokenizedSentence[idx]
            val tokenRange = tokensRanges[idx]
            tokens.add(
                Token(
                    text = token,
                    encodedData = encoder.encodeAsIds(token, false),
                    tokenRange = tokenRange, isFirst = idx == 0,
                    isUsed = isValidText(token)
                )
            )
        }

        return SentenceCorrections(sentId, sentence, tokens)
    }

    private fun isValidText(text: String): Boolean {
        return true
    }
}
