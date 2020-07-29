package org.jetbrains.research.kotlin.inference.extensions.primitives

fun map(array: FloatArray, lambda: (Float) -> Float, copy: Boolean): FloatArray {
    val actual = if (copy) FloatArray(array.size) else array
    for (i in actual.indices) actual[i] = lambda(array[i])

    return actual
}

fun map(array: DoubleArray, lambda: (Double) -> Double, copy: Boolean): DoubleArray {
    val actual = if (copy) DoubleArray(array.size) else array
    for (i in actual.indices) actual[i] = lambda(array[i])

    return actual
}

fun map(array: IntArray, lambda: (Int) -> Int, copy: Boolean): IntArray {
    val actual = if (copy) IntArray(array.size) else array
    for (i in actual.indices) actual[i] = lambda(array[i])

    return actual
}

fun map(array: ShortArray, lambda: (Short) -> Short, copy: Boolean): ShortArray {
    val actual = if (copy) ShortArray(array.size) else array
    for (i in actual.indices) actual[i] = lambda(array[i])

    return actual
}

fun map(array: LongArray, lambda: (Long) -> Long, copy: Boolean): LongArray {
    val actual = if (copy) LongArray(array.size) else array
    for (i in actual.indices) actual[i] = lambda(array[i])

    return actual
}

