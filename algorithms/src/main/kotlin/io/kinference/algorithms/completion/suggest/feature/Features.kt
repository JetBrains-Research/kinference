package io.kinference.algorithms.completion.suggest.feature

import io.kinference.algorithms.completion.generation.GenerationInfo


object Features {
    fun prob(generationInfo: GenerationInfo): Double {
        return generationInfo.probs.reduce(Double::times)
    }

    fun meanProb(generationInfo: GenerationInfo): Double {
        val probsSum = generationInfo.probs.reduce(Double::plus)
        return probsSum / generationInfo.probs.size
    }

    fun prefixMatchedCount(prefix: String, completion: String): Int {
        for (i in prefix.indices) {
            if (i == completion.length || prefix[i] != completion[i]) {
                return i
            }
        }
        return prefix.length
    }
}
