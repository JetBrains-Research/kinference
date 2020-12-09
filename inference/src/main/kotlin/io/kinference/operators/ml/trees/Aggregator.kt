package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.tiled.FloatTiledArray

//TODO: AVERAGE, MIN, MAX
//TODO: implement for multi-weight nodes
sealed class Aggregator {
    abstract fun accept(score: Float, value: Float): Float
    abstract fun finalize(base: Float, dst: FloatTiledArray, position: Int, score: Float): Float

    companion object {
        operator fun get(name: String) = when (name) {
            "SUM" -> Sum
            else -> error("")
        }
    }
}

object Sum : Aggregator() {
    override fun accept(score: Float, value: Float): Float {
        return score + value
    }
    override fun finalize(base: Float, dst: FloatTiledArray, position: Int, score: Float): Float {
        val newScore = base + score
        dst[position] = newScore
        return newScore
    }
}
