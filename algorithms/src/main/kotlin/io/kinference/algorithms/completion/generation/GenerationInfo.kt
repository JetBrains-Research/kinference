package io.kinference.algorithms.completion.generation

import kotlin.math.max
import kotlin.math.min

/**
 * Information regarding specific completion -- probabilities and length in words.
 */
class GenerationInfo(initProbs: List<Double> = ArrayList(), var wordLen: Int = 0) {
    /**
     * Probabilities of BPE tokens one by one
     *
     * Note, that probability of the whole thing is a multiplication of all probabilities in [probs]
     */
    var probs: DoubleArray = initProbs.toDoubleArray()
        private set

    internal fun add(prob: Double) {
        val newProbs = DoubleArray(probs.size + 1)
        probs.copyInto(newProbs)
        newProbs[newProbs.size - 1] = prob
        probs = newProbs
    }

    internal fun trim(left: Int, right: Int? = null): GenerationInfo {
        var realLeft = left
        var realRight: Int
        if (right == null) {
            realLeft = 0
            realRight = min(left, probs.size)
        } else {
            realRight = right
        }
        realLeft = max(0, min(realLeft, probs.size))
        realRight = max(0, min(realRight, probs.size))
        probs = probs.copyOfRange(realLeft, realRight)
        return this
    }
}
