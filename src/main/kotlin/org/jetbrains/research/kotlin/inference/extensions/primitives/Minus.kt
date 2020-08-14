package org.jetbrains.research.kotlin.inference.extensions.primitives

fun minus(left: FloatArray, leftOffset: Int, right: FloatArray, rightOffset: Int, destination: FloatArray, destinationOffset: Int, size: Int): FloatArray {
    require(left.size - leftOffset >= size)
    require(right.size - rightOffset >= size)
    require(destination.size - destinationOffset >= size)

    for (i in 0 until size) destination[destinationOffset + i] = left[leftOffset + i] - right[rightOffset + i]

    return destination
}

fun minus(left: IntArray, leftOffset: Int, right: IntArray, rightOffset: Int, destination: IntArray, destinationOffset: Int, size: Int): IntArray {
    require(left.size - leftOffset >= size)
    require(right.size - rightOffset >= size)
    require(destination.size - destinationOffset >= size)

    for (i in 0 until size) destination[destinationOffset + i] = left[leftOffset + i] - right[rightOffset + i]

    return destination
}

fun minus(left: LongArray, leftOffset: Int, right: LongArray, rightOffset: Int, destination: LongArray, destinationOffset: Int, size: Int): LongArray {
    require(left.size - leftOffset >= size)
    require(right.size - rightOffset >= size)
    require(destination.size - destinationOffset >= size)

    for (i in 0 until size) destination[destinationOffset + i] = left[leftOffset + i] - right[rightOffset + i]

    return destination
}

fun minus(left: DoubleArray, leftOffset: Int, right: DoubleArray, rightOffset: Int, destination: DoubleArray, destinationOffset: Int, size: Int): DoubleArray {
    require(left.size - leftOffset >= size)
    require(right.size - rightOffset >= size)
    require(destination.size - destinationOffset >= size)

    for (i in 0 until size) destination[destinationOffset + i] = left[leftOffset + i] - right[rightOffset + i]

    return destination
}

fun minus(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
    require(left.size - leftOffset >= size)
    require(right.size - rightOffset >= size)
    require(destination.size - destinationOffset >= size)

    for (i in 0 until size) destination[destinationOffset + i] = (left[leftOffset + i] - right[rightOffset + i]).toShort()

    return destination
}
