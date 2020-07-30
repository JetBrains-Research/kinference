package org.jetbrains.research.kotlin.inference.extensions.primitives

fun map(array: FloatArray, lambda: FloatArrayToFloatArray, copy: Boolean): FloatArray {
    val actual = if (copy) array.copyOf() else array
    return lambda.apply(actual)
}

fun map(array: DoubleArray, lambda: DoubleArrayToDoubleArray, copy: Boolean): DoubleArray {
    val actual = if (copy) array.copyOf() else array
    return lambda.apply(actual)
}

fun map(array: IntArray, lambda: IntArrayToIntArray, copy: Boolean): IntArray {
    val actual = if (copy) array.copyOf() else array
    return lambda.apply(actual)
}

fun map(array: ShortArray, lambda: ShortArrayToShortArray, copy: Boolean): ShortArray {
    val actual = if (copy) array.copyOf() else array
    return lambda.apply(actual)
}

fun map(array: LongArray, lambda: LongArrayToLongArray, copy: Boolean): LongArray {
    val actual = if (copy) array.copyOf() else array
    return lambda.apply(actual)
}

