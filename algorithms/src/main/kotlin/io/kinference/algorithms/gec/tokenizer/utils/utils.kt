package io.kinference.algorithms.gec.tokenizer.utils

object ControlChars{
    val values = setOf(CharCategory.UNASSIGNED, CharCategory.CONTROL,
        CharCategory.FORMAT, CharCategory.PRIVATE_USE, CharCategory.SURROGATE)
}

object PunctuationChars{
    val values = setOf(CharCategory.DASH_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION,
        CharCategory.END_PUNCTUATION, CharCategory.START_PUNCTUATION, CharCategory.OTHER_PUNCTUATION)
}

fun whitespaceTokenize(text: String): List<String>{
    val trimmed = text.trimStart().trimEnd()
    if (trimmed == ""){
        return emptyList()
    }
    return trimmed.split(" ")
}

fun isWhitespace(char: Char): Boolean{
    return char.isWhitespace()
}

fun isControl(char: Char): Boolean{
    if (char == '\t' || char == '\n' || char == '\r'){
        return false
    }
    if (char.category in ControlChars.values){
        return true
    }
    return false
}

fun isPunctuation(char: Char): Boolean{
    val cp = char.toInt()

    if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)){
        return true
    }
    if (char.category in PunctuationChars.values){
        return true
    }
    return false
}
