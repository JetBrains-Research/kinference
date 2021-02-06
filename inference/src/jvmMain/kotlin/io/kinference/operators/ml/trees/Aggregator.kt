package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.tiled.FloatTiledArray

//TODO: AVERAGE, MIN, MAX
//TODO: implement for multi-weight nodes
sealed class Aggregator {
    abstract fun accept(score: FloatArray, value: FloatArray, startIdx: Int = 0): FloatArray
    abstract fun finalize(base: FloatArray, dst: FloatTiledArray, position: Int, score: FloatArray, numTargets: Int = 1): FloatArray

    companion object {
        operator fun get(name: String) = when (name) {
            "SUM" -> Sum
            else -> error("Unsupported aggregation function: $name")
        }
    }
}

object Sum : Aggregator() {
    override fun accept(score: FloatArray, value: FloatArray, startIdx: Int): FloatArray {
        for (i in score.indices) score[i] += value[i + startIdx]
        return score
    }

    override fun finalize(base: FloatArray, dst: FloatTiledArray, position: Int, score: FloatArray, numTargets: Int): FloatArray {
        for (i in 0 until numTargets) {
            score[i] = base[i] + score[i]
            dst[i + position] = score[i]
        }
        return score
    }
}
