package io.kinference.tfjs.operators.ml.trees

//TODO: AVERAGE, MIN, MAX
//TODO: implement for multi-weight nodes

enum class AggregatorType {
    SUM,
    AVERAGE,
    MIN,
    MAX
}

sealed class Aggregator {
    abstract fun accept(score: FloatArray, value: FloatArray, startIdx: Int = 0): FloatArray
    abstract fun finalize(base: FloatArray, dst: FloatArray, dstPosition: Int, score: FloatArray, numTargets: Int = 1): FloatArray

    companion object {
        operator fun get(name: AggregatorType) = when (name) {
            AggregatorType.SUM -> Sum
            else -> error("Unsupported aggregation function: $name")
        }
    }
}

object Sum : Aggregator() {
    override fun accept(score: FloatArray, value: FloatArray, startIdx: Int): FloatArray {
        for (i in score.indices) score[i] += value[startIdx + i]
        return score
    }

    override fun finalize(base: FloatArray, dst: FloatArray, dstPosition: Int, score: FloatArray, numTargets: Int): FloatArray {
        for (i in 0 until numTargets) {
            score[i] += base[i]
            dst[dstPosition + i] = score[i]
        }
        return score
    }
}
