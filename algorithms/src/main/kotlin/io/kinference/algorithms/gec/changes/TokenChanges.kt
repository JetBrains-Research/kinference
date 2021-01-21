package io.kinference.algorithms.gec.changes

/**
 * Changes of token for every correction iteration
 */
data class TokenChanges(
    val replacement: String,
    var tokenizedReplacement: List<String>? = listOf(replacement),
    val usedTokensNum: Int = 1
)
