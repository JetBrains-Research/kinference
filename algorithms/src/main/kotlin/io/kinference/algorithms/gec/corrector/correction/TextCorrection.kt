package io.kinference.algorithms.gec.corrector.correction

/**
 * Correction of error in text -- output of GEC model.
 *
 * @param errorRange is a range for which replacement is prepared
 * @param underlineRange is a range that should be highlighted
 * @param replacement is a suggestion of fix
 * @param message is a description of fix
 */
data class TextCorrection(val errorRange: IntRange,
                          val underlineRange: IntRange,
                          val replacement: String,
                          val message: String) {
    /** Apply correction to this sentence */
    fun apply(sentence: String): String = sentence.replaceRange(errorRange, replacement)
}
