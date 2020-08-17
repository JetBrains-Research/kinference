package org.jetbrains.research.kotlin.inference.extensions.primitives

fun minus(array: FloatArray, scalar: Number, copy: Boolean): FloatArray {
    val result = if (copy) FloatArray(array.size) else array

    val value = scalar.toFloat()

    for (i in array.indices) result[i] = array[i] - value

    return result
}

fun minus(array: IntArray, scalar: Number, copy: Boolean): IntArray {
    val result = if (copy) IntArray(array.size) else array

    val value = scalar.toInt()

    for (i in array.indices) result[i] = array[i] - value

    return result
}

fun minus(array: LongArray, scalar: Number, copy: Boolean): LongArray {
    val result = if (copy) LongArray(array.size) else array

    val value = scalar.toLong()

    for (i in array.indices) result[i] = array[i] - value

    return result
}

fun minus(array: DoubleArray, scalar: Number, copy: Boolean): DoubleArray {
    val result = if (copy) DoubleArray(array.size) else array

    val value = scalar.toDouble()

    for (i in array.indices) result[i] = array[i] - value

    return result
}

fun minus(array: ShortArray, scalar: Number, copy: Boolean): ShortArray {
    val result = if (copy) ShortArray(array.size) else array

    val value = scalar.toShort()

    for (i in array.indices) result[i] = (array[i] - value).toShort()

    return result
}
