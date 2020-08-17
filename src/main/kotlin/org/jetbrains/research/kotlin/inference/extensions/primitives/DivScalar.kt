package org.jetbrains.research.kotlin.inference.extensions.primitives

fun div(array: FloatArray, offset: Int, scalar: Float, destination: FloatArray, destinationOffset: Int, size: Int): FloatArray {
    for (i in 0 until size) destination[destinationOffset + i] = array[offset + i] / scalar

    return destination
}

fun div(array: IntArray, offset: Int, scalar: Int, destination: IntArray, destinationOffset: Int, size: Int): IntArray {
    for (i in 0 until size) destination[destinationOffset + i] = array[offset + i] / scalar

    return destination
}

fun div(array: LongArray, offset: Int, scalar: Long, destination: LongArray, destinationOffset: Int, size: Int): LongArray {
    for (i in 0 until size) destination[destinationOffset + i] = array[offset + i] / scalar

    return destination
}

fun div(array: DoubleArray, offset: Int, scalar: Double, destination: DoubleArray, destinationOffset: Int, size: Int): DoubleArray {
    for (i in 0 until size) destination[destinationOffset + i] = array[offset + i] / scalar

    return destination
}

fun div(array: ShortArray, offset: Int, scalar: Short, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
    for (i in 0 until size) destination[destinationOffset + i] = (array[offset + i] / scalar).toShort()

    return destination
}
