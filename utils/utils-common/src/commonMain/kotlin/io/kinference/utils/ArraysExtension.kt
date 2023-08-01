package io.kinference.utils

fun LongArray.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun UByteArray.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun UShortArray.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun UIntArray.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun ULongArray.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun Array<Long>.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun Array<UByte>.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun Array<UShort>.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun Array<UInt>.toFloatArray() = FloatArray(size) { this[it].toFloat() }
fun Array<ULong>.toFloatArray() = FloatArray(size) { this[it].toFloat() }

fun LongArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun UByteArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun UShortArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun UIntArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun ULongArray.toIntArray() = IntArray(size) { this[it].toInt() }
fun Array<Long>.toIntArray() = IntArray(size) { this[it].toInt() }
fun Array<UByte>.toIntArray() = IntArray(size) { this[it].toInt() }
fun Array<UShort>.toIntArray() = IntArray(size) { this[it].toInt() }
fun Array<UInt>.toIntArray() = IntArray(size) { this[it].toInt() }
fun Array<ULong>.toIntArray() = IntArray(size) { this[it].toInt() }

fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }

fun Collection<Number>.toIntArray(): IntArray {
    val array = IntArray(this.size)
    for ((i, element) in this.withIndex()) {
        array[i] = element.toInt()
    }
    return array
}

fun IntProgression.toIntArray(): IntArray {
    val iter = iterator()

    return IntArray(count()) { iter.next() }
}

fun IntProgression.toTypedIntArray(): Array<Int> {
    val iter = iterator()

    return Array(count()) { iter.next() }
}
