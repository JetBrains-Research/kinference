package io.kinference.ndarray

import io.kinference.ndarray.arrays.Strides
import io.kinference.utils.launchWithLimitOrDefault
import kotlinx.coroutines.coroutineScope
import kotlin.math.min

fun Double.toUShort() = this.toInt().toUShort()
fun Double.toUByte() = this.toInt().toUByte()

fun IntRange.reversed(): IntArray {
    val size = this.last - this.first + 1
    val array = IntArray(size)
    for ((i, element) in this.withIndex()) {
        array[size - i - 1] = element
    }
    return array
}

fun Int.concat(array: IntArray): IntArray {
    val copy = IntArray(array.size + 1)
    array.copyInto(copy, 1)
    copy[0] = this
    return copy
}

fun IntArray.concat(value: Int): IntArray {
    val copy = IntArray(size + 1)
    this.copyInto(copy)
    copy[size] = value
    return copy
}

private const val MIN_BLOCK_SIZE = 512

fun blockSizeByStrides(strides: Strides): Int {
    return when {
        strides.linearSize == 0 -> 0
        strides.shape.isEmpty() -> 1
        else -> {
            val rowSize = strides.shape.last()

            val blockSize = if (rowSize < MIN_BLOCK_SIZE) rowSize else {
                var num = rowSize / MIN_BLOCK_SIZE
                while (rowSize % num != 0) num--
                rowSize / num
            }

            blockSize
        }
    }
}

const val ERF_P_VALUE = 0.3275911
const val ERF_P_VALUE_FLOAT = 0.3275911f
val ERF_COEF = doubleArrayOf(
    0.254829592,
    -0.284496736,
    1.421413741,
    -1.453152027,
    1.061405429
)

const val ERF_COEF_1_FLOAT = 0.254829592f
const val ERF_COEF_2_FLOAT = -0.284496736f
const val ERF_COEF_3_FLOAT = 1.421413741f
const val ERF_COEF_4_FLOAT = -1.453152027f
const val ERF_COEF_5_FLOAT = 1.061405429f

const val ERF_COEF_1 = 0.254829592
const val ERF_COEF_2 = -0.284496736
const val ERF_COEF_3 = 1.421413741
const val ERF_COEF_4 = -1.453152027
const val ERF_COEF_5 = 1.061405429

internal fun IntArray.swap(leftIdx: Int, rightIdx: Int) {
    val temp = get(leftIdx)
    this[leftIdx] = this[rightIdx]
    this[rightIdx] = temp
}
/*
 * Parallelize with batching by minDataPerLaunch
 */
suspend fun parallelizeByBlocks(blockSize: Int,
                                countBlocks: Int,
                                minDataPerLaunch: Int,
                                body: (blockStart: Int, blockEnd: Int, coroutineIndex: Int) -> Unit) {

    val batchSize = batchSizeByData(blockSize, countBlocks, minDataPerLaunch)

    if (batchSize == countBlocks) {
        body(0, countBlocks, 0)
    } else {
        coroutineScope {
            for ((index, blockStart) in (0 until countBlocks step batchSize).withIndex()) {
                launchWithLimitOrDefault {
                    body(blockStart, min(blockStart + batchSize, countBlocks), index)
                }
            }
        }
    }
}

suspend inline fun parallelizeByRows(rowSize: Int, countRows: Int, minDataPerLaunch: Int, noinline body: (rowStart: Int, rowEnd: Int, index: Int) -> Unit) = parallelizeByBlocks(rowSize, countRows, minDataPerLaunch, body)

internal fun countCoroutinesByData(rowSize: Int, countRows: Int, minDataPerLaunch: Int): Int {
    val batchSize = batchSizeByData(rowSize, countRows, minDataPerLaunch)

    return (countRows + batchSize - 1) / batchSize
}

internal fun batchSizeByData(rowSize: Int, countRows: Int, minDataPerLaunch: Int): Int {
    val batchSize = (minDataPerLaunch + rowSize - 1) / rowSize

    return min(batchSize, countRows)
}
