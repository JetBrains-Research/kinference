package io.kinference.trees

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
    abstract fun finalize(base: FloatArray?, dst: FloatArray, score: FloatArray, dstPosition: Int = 0, numTargets: Int = 1): FloatArray

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

    override fun finalize(base: FloatArray?, dst: FloatArray, score: FloatArray, dstPosition: Int, numTargets: Int): FloatArray {
        if (base != null) {
            for (i in 0 until numTargets) {
                score[i] += base[i]
                dst[dstPosition + i] = score[i]
            }
        } else {
            for (i in 0 until numTargets) {
                dst[dstPosition + i] = score[i]
            }
        }
        return score
    }
}
