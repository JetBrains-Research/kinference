package io.kinference.ndarray

import io.kinference.ndarray.arrays.Strides

fun Double.toUShort() = this.toInt().toUShort()
fun Double.toUByte() = this.toInt().toUByte()

fun Collection<Number>.toIntArray(): IntArray {
    val array = IntArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toInt()
    }
    return array
}

fun LongArray.toIntArray() = IntArray(this.size) { this[it].toInt() }
fun IntArray.toByteArray() = ByteArray(this.size) { this[it].toByte() }
fun IntArray.toUByteArray() = UByteArray(this.size) { this[it].toUByte() }
fun IntArray.toBooleanArray() = BooleanArray(this.size) { this[it] != 0 }
fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }

fun Collection<Number>.toFloatArray(): FloatArray {
    val array = FloatArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toFloat()
    }
    return array
}

fun IntRange.reversed(): IntArray {
    val size = this.last - this.first + 1
    val array = IntArray(size)
    for ((i, element) in this.withIndex()) {
        array[size - i - 1] = element
    }
    return array
}

fun IntRange.toIntArray(): IntArray {
    val size = this.last - this.first + 1
    val array = IntArray(size)
    for ((i, element) in this.withIndex()) {
        array[i] = element
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
