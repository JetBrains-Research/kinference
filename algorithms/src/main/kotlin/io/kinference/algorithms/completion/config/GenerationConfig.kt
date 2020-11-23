package io.kinference.algorithms.completion.config

data class GenerationConfig(
    val minLen: Int,
    val maxLen: Int,
    val numBeams: Int,
    val numGroups: Int,
    val repetitionPenalty: Double,
    val lengthPenalty: Double,
    val lenNormBase: Double,
    val lenNormPow: Double,
    val prefixErrLimit: Int,
    val spellProb: Double
) {

    companion object {
        val default = GenerationConfig(
            minLen = 1,
            maxLen = 3,
            numBeams = 5,
            numGroups = 1,
            repetitionPenalty = 1.0,
            lengthPenalty = 1.0,
            lenNormBase = 5.0,
            lenNormPow = 0.7,
            prefixErrLimit = 0,
            spellProb = 0.0001
        )
    }
}
