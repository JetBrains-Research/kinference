package org.jetbrains.research.kotlin.inference.extensions.primitives

fun minus(left: FloatArray, right: FloatArray, copy: Boolean): FloatArray {
    return plus(left, -right, copy)
}

fun minus(left: IntArray, right: IntArray, copy: Boolean): IntArray {
    return plus(left, -right, copy)
}

fun minus(left: LongArray, right: LongArray, copy: Boolean): LongArray {
    return plus(left, -right, copy)
}

fun minus(left: DoubleArray, right: DoubleArray, copy: Boolean): DoubleArray {
    return plus(left, -right, copy)
}

fun minus(left: ShortArray, right: ShortArray, copy: Boolean): ShortArray {
    return plus(left, -right, copy)
}
