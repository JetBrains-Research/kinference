package io.kinference.algorithms.gec.pretokenizer

class TokenSplits{

    val prefixes = ArrayList<String>()
    val suffixes = ArrayList<String>()
    var word: String? = null
    val wordTokens = ArrayList<String>()
    var isSpecial = false

    fun toList(): List<String> {
        return prefixes + wordTokens + suffixes.reversed()
    }
}
