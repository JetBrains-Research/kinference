package io.kinference.algorithms.completion.generation

import kotlin.math.max
import kotlin.math.min

class GenerationInfo(initProbs: List<Double> = ArrayList(), var score: Double = -1000.0, var wordLen: Int = 0) {
    var probs: DoubleArray
        private set

    init {
        probs = initProbs.toDoubleArray()
    }

    fun add(prob: Double) {
        val newProbs = DoubleArray(probs.size + 1)
        probs.copyInto(newProbs)
        newProbs[newProbs.size - 1] = prob
        probs = newProbs
    }

    fun trim(left: Int, right: Int? = null): GenerationInfo {
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
