package io.kinference.algorithms.completion.generating

import java.lang.IllegalArgumentException
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.*

fun IntRange.toLongArray(): LongArray {
    val arr = LongArray(last - first + 1)
    this.forEachIndexed { i, item -> arr[i] = item.toLong() }
    return arr
}

fun Array<IntArray>.toLongArray(): LongArray {
    val arr = LongArray(this.sumBy { it.size })
    var off = 0
    for(block in this) {
        for (value in block) arr[off++] = value.toLong()
    }
    return arr
}

fun IntArray.sliceArray(indices: IntArray): IntArray {
    val result = IntArray(indices.size)
    var targetIndex = 0
    for (sourceIndex in indices) {
        result[targetIndex++] = this[sourceIndex]
    }
    return result
}

fun DoubleArray.sliceArray(indices: IntArray): DoubleArray {
    val result = DoubleArray(indices.size)
    var targetIndex = 0
    for (sourceIndex in indices) {
        result[targetIndex++] = this[sourceIndex]
    }
    return result
}

fun <T> List<T>.slice(indices: IntArray): List<T> {
    val result = ArrayList<T>(indices.size)
    for ((targetIndex, sourceIndex) in indices.withIndex()) {
        result.add(targetIndex, this[sourceIndex])
    }
    return result
}

fun logSoftmax(scores: Array<DoubleArray>): Array<DoubleArray> {
    val expScores = Array(scores.size) {
        val curScores = scores[it]
        DoubleArray(curScores.size) { i -> exp(curScores[i]) }
    }
    for (score in expScores) {
        val scoresSum = score.sum()
        for (i in score.indices) score[i] = ln(score[i] / scoresSum)
    }
    return expScores
}

fun topk1d(data: DoubleArray, size: Int): IntArray {
    val pairedData = Array(data.size) { Pair(data[it], it) }
    Arrays.parallelSort(pairedData) { fst: Pair<Double, Int>, snd: Pair<Double, Int> -> -fst.first.compareTo(snd.first) }
    return IntArray(size) { pairedData[it].second }
}

fun topk2d(data: Array<DoubleArray>, size: Int, dim: Int = 0): Array<IntArray> {
    if (data.isEmpty()) {
        return emptyArray()
    }

    when (dim) {
        0 -> {
            val listSize = min(data.size, size)
            val result = Array(listSize) { IntArray(data[0].size) }
            for (j in data[0].indices) {
                val slice = DoubleArray(data.size) { data[it][j] }
                val topColumn = topk1d(slice, size)
                for (i in topColumn.indices) result[i][j] = topColumn[i]
            }
            return result
        }
        1 -> {
            return Array(data.size) { topk1d(data[it], size) }
        }
        else -> {
            throw IllegalArgumentException("Index should be 0 or 1")
        }
    }
}
