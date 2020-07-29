package org.jetbrains.research.kotlin.inference.extensions.primitives

fun minus(left: FloatArray, right: FloatArray): FloatArray {
    return plus(left, -right, true)
}

fun minus(left: IntArray, right: IntArray): IntArray {
    return plus(left, -right, true)
}

fun minus(left: LongArray, right: LongArray): LongArray {
    return plus(left, -right, true)
}

fun minus(left: DoubleArray, right: DoubleArray): DoubleArray {
    return plus(left, -right, true)
}

fun minus(left: ShortArray, right: ShortArray): ShortArray {
    return plus(left, -right, true)
}
