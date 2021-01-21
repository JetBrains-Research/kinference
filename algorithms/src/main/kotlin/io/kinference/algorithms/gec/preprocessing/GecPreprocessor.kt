package io.kinference.algorithms.gec.preprocessing

import io.kinference.algorithms.gec.encoder.PreTrainedTextEncoder
import io.kinference.algorithms.gec.tokenizer.utils.tokenizeByWhitespace
import io.kinference.algorithms.gec.utils.SentenceCorrections
import io.kinference.algorithms.gec.utils.GECToken
import io.kinference.algorithms.gec.utils.GECToken.TokenRange
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

        val tokens = ArrayList<GECToken>()
        if (useStartToken) {
            tokens.add(
                GECToken(
                    text = "\$START", encoded = encoder.encodeAsIds("\$START", false),
                    range = TokenRange(start = 0, end = 0, withSpace = false), isFirst = false, isUsed = true
                )
            )
        }
        for (idx in tokenizedSentence.indices) {
            val token = tokenizedSentence[idx]
            val tokenRange = tokensRanges[idx]
            tokens.add(
                GECToken(
                    text = token,
                    encoded = encoder.encodeAsIds(token, false),
                    range = tokenRange, isFirst = idx == 0,
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
