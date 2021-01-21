package io.kinference.algorithms.gec.tokenizer.utils

/**
 * Class for char classification for preprocessing text data
 */
object CharUtils {
    private val controlChars = setOf(CharCategory.UNASSIGNED, CharCategory.CONTROL,
        CharCategory.FORMAT, CharCategory.PRIVATE_USE, CharCategory.SURROGATE)

    private val punctuationChars = setOf(CharCategory.DASH_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION,
        CharCategory.END_PUNCTUATION, CharCategory.START_PUNCTUATION, CharCategory.OTHER_PUNCTUATION)

    fun isPunctuation(char: Char): Boolean {
        if (char.category in punctuationChars){
            return true
        }
        //Punctuation symbols from ASCII
        if ((char.toInt() in 33..47) || (char.toInt() in 58..64) || (char.toInt() in 91..96) || (char.toInt() in 123..126)){
            return true
        }
        return false
    }

    fun isControl(char: Char): Boolean {
        if (char == '\t' || char == '\n' || char == '\r') {
            return false
        }
        if (char.category in controlChars) {
            return true
        }
        return false
    }
}
/** Tokenize text by whitespaces removing blank parts */
fun String.tokenizeByWhitespace(): List<String> = split(" ").filter { it.isNotBlank() }

