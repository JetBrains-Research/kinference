package io.kinference.algorithms.gec.utils

import io.kinference.algorithms.gec.preprocessing.VerbsFormVocabulary



fun offsetCalc(sentIds: List<List<Int>>, offsetType: String): List<Int> {
    val wordLens = sentIds.map { it.size }
    if (offsetType == "first") {
        var acc = 1
        val tokenPlaceIdx = mutableListOf(acc)
        for (wordLen in wordLens.dropLast(1)) {
            acc += wordLen
            tokenPlaceIdx.add(acc)
        }
        return tokenPlaceIdx

    } else {
        throw NotImplementedError("Not implemented error")
    }
}
