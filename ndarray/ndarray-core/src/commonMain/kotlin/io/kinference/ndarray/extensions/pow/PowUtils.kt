package io.kinference.ndarray.extensions.pow

import io.kinference.ndarray.arrays.*
import kotlin.math.pow

internal fun Float.pow(x: Double): Float = this.toDouble().pow(x).toFloat()
internal fun Int.pow(x: Double): Int = this.toDouble().pow(x).toInt()
internal fun Long.pow(x: Double): Long = this.toDouble().pow(x).toLong()

private fun FloatNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun IntNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun UIntNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun ByteNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun UByteNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun ShortNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

private fun UShortNDArray.toDoubleNDArray(): DoubleNDArray {
    val pointer = array.pointer()
    return DoubleNDArray(shape) { pointer.getAndIncrement().toDouble() }
}

internal fun NumberNDArrayCore.toDoubleNDArray(): DoubleNDArray {
    return when (this) {
        is DoubleNDArray -> this
        is FloatNDArray -> toDoubleNDArray()
        is IntNDArray -> toDoubleNDArray()
        is UIntNDArray -> toDoubleNDArray()
        is ByteNDArray -> toDoubleNDArray()
        is UByteNDArray -> toDoubleNDArray()
        is ShortNDArray -> toDoubleNDArray()
        is UShortNDArray -> toDoubleNDArray()
        else -> error("Unsupported array data type: ${this.type}")
    }
}
