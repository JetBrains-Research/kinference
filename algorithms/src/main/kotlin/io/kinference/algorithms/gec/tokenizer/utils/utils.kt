package io.kinference.algorithms.gec.tokenizer.utils

object CharUtils {
    private val controlChars = setOf(CharCategory.UNASSIGNED, CharCategory.CONTROL,
        CharCategory.FORMAT, CharCategory.PRIVATE_USE, CharCategory.SURROGATE)

    private val punctuationChars = setOf(CharCategory.DASH_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION,
        CharCategory.END_PUNCTUATION, CharCategory.START_PUNCTUATION, CharCategory.OTHER_PUNCTUATION)

    fun isPunctuation(category: CharCategory): Boolean {
        return category in punctuationChars
    }

    fun isControl(category: CharCategory): Boolean {
        return category in controlChars
    }
}

fun whitespaceTokenize(text: String): List<String> {
    val trimmed = text.trimStart().trimEnd()
    if (trimmed == "") {
        return emptyList()
    }
    return trimmed.split(" ")
}

fun isWhitespace(char: Char): Boolean {
    return char.isWhitespace()
}

fun isControl(char: Char): Boolean {
    if (char == '\t' || char == '\n' || char == '\r') {
        return false
    }
    if (CharUtils.isControl(char.category)) {
        return true
    }
    return false
}

fun isPunctuation(char: Char): Boolean {
    val cp = char.toInt()

    if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)) {
        return true
    }
    if (CharUtils.isPunctuation(char.category)) {
        return true
    }
    return false
}
