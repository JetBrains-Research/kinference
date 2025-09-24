package io.kinference.ndarray

import io.kinference.ndarray.arrays.Strides
import io.kinference.utils.launchWithLimitOrDefault
import kotlinx.coroutines.coroutineScope
import kotlin.math.min
import oshi.SystemInfo
import oshi.hardware.CentralProcessor

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

private val L1CacheSize: Int? = try {
    val processor = SystemInfo().hardware.processor
    val caches = processor.processorCaches.filter {
        it.level.toInt() == 1 && it.type == CentralProcessor.ProcessorCache.Type.DATA
    }
    if (caches.size != 1) null else caches[0].cacheSize
} catch (e: Exception) {
    null
}

private const val MIN_BLOCK_SIZE = 1024
private val MAX_BLOCK_SIZE = when (L1CacheSize) {
    null -> MIN_BLOCK_SIZE
    else -> maxOf(L1CacheSize * 3 / 32, MIN_BLOCK_SIZE)
}

private fun getBlocksize(size: Int): Int {
    if (size < MAX_BLOCK_SIZE) return size
    var blockNum = size / MAX_BLOCK_SIZE
    val maxBlockNum = size / MIN_BLOCK_SIZE
    while (size % blockNum != 0 && blockNum + 1 < maxBlockNum) blockNum++
    while (size % blockNum != 0) blockNum--
    return size / blockNum
}

fun blockSizeByStrides(strides: Strides): Int {
    return when {
        strides.linearSize == 0 -> 0
        strides.shape.isEmpty() -> 1
        else -> getBlocksize(strides.shape.last())
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

const val INIT_STORAGE_SIZE = 64

internal fun IntArray.swap(leftIdx: Int, rightIdx: Int) {
    val temp = get(leftIdx)
    this[leftIdx] = this[rightIdx]
    this[rightIdx] = temp
}

fun interface ParallelizeBody {
    operator fun invoke(start: Int, end: Int, coroutineIndex: Int)
}

/*
 * Parallelize with batching by minDataPerLaunch
 */
suspend fun parallelizeByBlocks(
    blockSize: Int,
    countBlocks: Int,
    minDataPerLaunch: Int,
    body: ParallelizeBody
) {

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

suspend inline fun parallelizeByRows(rowSize: Int, countRows: Int, minDataPerLaunch: Int, body: ParallelizeBody) =
    parallelizeByBlocks(rowSize, countRows, minDataPerLaunch, body)

internal fun countCoroutinesByData(rowSize: Int, countRows: Int, minDataPerLaunch: Int): Int {
    val batchSize = batchSizeByData(rowSize, countRows, minDataPerLaunch)

    return (countRows + batchSize - 1) / batchSize
}

internal fun batchSizeByData(rowSize: Int, countRows: Int, minDataPerLaunch: Int): Int {
    val batchSize = (minDataPerLaunch + rowSize - 1) / rowSize

    return min(batchSize, countRows)
}
