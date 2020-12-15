package io.kinference.algorithms.gec.tokenizer.utils

fun whitespaceTokenize(text: String): List<String>{
    val trimmed = text.trimStart().trimEnd()
    if (trimmed == ""){
        return emptyList()
    }
    return trimmed.split(" ")
}

fun isWhitespace(char: Char): Boolean{
    if (char.isWhitespace()){
        return true
    }
    if (char.category == CharCategory.SPACE_SEPARATOR){
        return true
    }
    return false
}

fun isControl(char: Char): Boolean{
    if (char == '\t' || char == '\n' || char == '\r'){
        return false
    }
    if (char.category in listOf(CharCategory.UNASSIGNED, CharCategory.CONTROL,
            CharCategory.FORMAT, CharCategory.PRIVATE_USE, CharCategory.SURROGATE)){
        return true
    }
    return false
}

fun isPunctuation(char: Char): Boolean{
    val cp = char.toInt()

    if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)){
        return true
    }
    if (char.category in listOf(CharCategory.DASH_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION,
            CharCategory.END_PUNCTUATION, CharCategory.START_PUNCTUATION, CharCategory.OTHER_PUNCTUATION)){
        return true
    }
    return false
}
