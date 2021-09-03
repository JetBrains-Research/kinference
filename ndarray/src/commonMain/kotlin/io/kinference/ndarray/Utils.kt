package io.kinference.ndarray

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

fun IntArray?.isNullOrEmpty() = this == null || this.isEmpty()
fun LongArray?.isNullOrEmpty() = this == null || this.isEmpty()

fun Collection<Number>.toFloatArray(): FloatArray {
    val array = FloatArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toFloat()
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

const val ERF_P_VALUE = 0.3275911
val ERF_COEF = doubleArrayOf(
    0.254829592,
    -0.284496736,
    1.421413741,
    -1.453152027,
    1.061405429
)

internal fun IntArray.swap(leftIdx: Int, rightIdx: Int) {
    val temp = get(leftIdx)
    this[leftIdx] = this[rightIdx]
    this[rightIdx] = temp
}
