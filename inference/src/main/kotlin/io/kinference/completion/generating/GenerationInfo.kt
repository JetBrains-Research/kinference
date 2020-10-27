package io.kinference.completion.generating

import java.lang.Integer.max
import java.lang.Integer.min

class GenerationInfo(initProbs: List<Double> = ArrayList(), var score: Double = -1000.0, var wordLen: Int = 0) {
    var probs: MutableList<Double>
        private set

    init {
        probs = initProbs.toMutableList()
    }

    fun add(prob: Double) {
        probs.add(prob)
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
        probs = probs.subList(realLeft, realRight)
        return this
    }
}
