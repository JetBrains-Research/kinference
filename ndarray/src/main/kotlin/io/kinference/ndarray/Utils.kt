package io.kinference.ndarray

import io.kinference.primitives.types.PrimitiveType

fun Double.toUShort() = this.toInt().toUShort()
fun Double.toUByte() = this.toInt().toUByte()

fun PrimitiveType.toFloat(): Float = throw UnsupportedOperationException()
fun PrimitiveType.toDouble(): Double = throw UnsupportedOperationException()
fun PrimitiveType.toInt(): Int = throw UnsupportedOperationException()

fun Collection<Number>.toIntArray(): IntArray {
    val array = IntArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toInt()
    }
    return array
}

fun IntProgression.toIntArray(): IntArray {
    val array = IntArray(this.count())
    for ((i, element) in this.withIndex()) {
        array[i] = element
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
    System.arraycopy(array, 0, copy, 1, array.size)
    copy[0] = this
    return copy
}

fun IntArray.concat(value: Int): IntArray {
    val copy = IntArray(size + 1)
    System.arraycopy(this, 0, copy, 0, size)
    copy[size] = value
    return copy
}

const val ERF_P_VALUE = 0.3275911
val ERF_COEF = doubleArrayOf(
    0.254829592,
    -0.284496736,
    1.421413741,
    -1.453152027,
    1.061405429
)
