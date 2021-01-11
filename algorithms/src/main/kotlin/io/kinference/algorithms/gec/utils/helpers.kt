package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary
import io.kinference.algorithms.gec.tokenizer.utils.isPunctuation

fun transformUsingVerb(token: String, form: String, verbsVocab: VerbsFormVocabulary): String {
    val formDict = verbsVocab.verbs2verbs[token]
    return if (formDict == null) {
        token
    } else {
        val verb = formDict[form]
        verb ?: token
    }
}

fun transformUsingSplit(token: String): List<String> {
    return token.split("-")
}

fun transformUsingCase(token: String, case: String): String {
    if (case == "LOWER") {
        return token.toLowerCase()
    } else if (case == "UPPER") {
        return token.toUpperCase()
    } else if (case == "CAPITAL") {
        return token.capitalize()
    } else if (case == "CAPITAL_1") {
        val first = token[0]
        val rst = token.substring(startIndex = 1).capitalize()
        return first + rst
    } else if (case == "UPPER_-1") {
        val last = token[-1]
        val rst = token.substring(startIndex = 0, endIndex = token.length - 1)
        return rst.toUpperCase() + last
    } else {
        return token
    }
}

fun offsetCalc(sentIds: List<List<Int>>, offsetType: String): List<Int> {
    val wordLens = sentIds.map { it.size }
    if (offsetType == "first") {
        var acc = 1
        val tokenPlaceIdx = mutableListOf<Int>(acc)
        for (wordLen in wordLens.dropLast(1)){
            acc += wordLen
            tokenPlaceIdx.add(acc)
        }
        return tokenPlaceIdx

    } else {
        throw NotImplementedError("Not implemented error")
    }
}

fun createMessageBasedOnTag(tag: String): String{
    if (tag == "\$TRANSFORM_CASE_CAPITAL"){
        return "Capitalized first letter"
    }
    else if(tag.startsWith("\$TRANSFORM_VERB")){
        return "Incorrect verb form"
    }
    else if(tag in listOf("\$TRANSFORM_AGREEMENT_PLURAL", "\$TRANSFORM_AGREEMENT_SINGULAR")){
        return "Incorrect word form"
    }
    else if(tag == "\$DELETE_SPACES"){
        return "Rudimentary spaces"
    }
    else if (tag.startsWith("\$APPEND_") or tag.startsWith("\$REPLACE_")){
        val last = tag.split('_').last()
        if (isPunctuation(last.toCharArray().last())){
            return "Incorrect punctuation"
        }
        else if(last in listOf("a", "an", "the")){
            return "Article error"
        }
        else{
            return "Grammatical error"
        }
    }
    else{
        return "Grammatical error"
    }
}

fun calculateTokensBordersAndWithSpaces(text: String, tokens: List<String>, textWithSpace: Boolean = false): List<TokenRange> {
    val result = ArrayList<TokenRange>()
    var startFrom = 0
    for ((idx, token) in tokens.withIndex()) {
        val startIdxAndString = text.findAnyOf(strings = listOf(token), startIndex = startFrom)!!

        val withSpace: Boolean
        assert(startIdxAndString.first != -1)
        if (idx == 0 && textWithSpace) {
            withSpace = true
        } else {
            withSpace = startIdxAndString.first >= startFrom + 1
        }
        result.add(TokenRange(start = startIdxAndString.first,
            end = startIdxAndString.first + token.length,
            withSpace = withSpace))
        startFrom = startIdxAndString.first + token.length

    }
    return result
}
