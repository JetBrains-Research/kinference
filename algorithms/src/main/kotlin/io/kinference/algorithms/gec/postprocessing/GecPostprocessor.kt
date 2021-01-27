package io.kinference.algorithms.gec.postprocessing

import io.kinference.algorithms.gec.corrector.correction.SentenceCorrections
import io.kinference.algorithms.gec.corrector.correction.TextCorrection
import io.kinference.algorithms.gec.utils.*


/**
 * Postprocessor class
 */
abstract class GecPostprocessor {
    abstract fun postprocess(sentObj: SentenceCorrections): String
}

/**
 * Basic realization of postprocessor
 */
class GecCorrectionPostprocessor : GecPostprocessor() {
    override fun postprocess(sentObj: SentenceCorrections): String {
        val original = sentObj.sent
        val textCorrections = sentObj.toTextCorrections()

        return transformSentence(original, textCorrections)
    }

    /**
     * function which create new sentence from incorrect sentence and TextCorrection list
     * @param sentence original sentence
     * @param corrections corrections of sentence
     */
    private fun transformSentence(sentence: String, corrections: List<TextCorrection>): String {
        var result = sentence
        var offset = 0
        for (correction in corrections) {
            val startEnd = correction.errorRange
            result = result.replaceRange(correction.errorRange.withOffset(offset), correction.replacement)
            offset += correction.replacement.length - (startEnd.endInclusive + 1 - startEnd.start)
        }
        return result
    }
}

class GecEvalPostprocessor : GecPostprocessor() {
    override fun postprocess(sentObj: SentenceCorrections): String {
        val tokens = sentObj.toCorrectedTokenSentence()
        val validTextTokens = tokens.map { it.text }.filter { it != "\$START" && it != "" }
        return validTextTokens.joinToString(" ")
    }
}
