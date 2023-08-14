package io.kinference.ndarray.math

inline fun Math.floorMod(left: Byte, right: Byte) = floorMod(left.toInt(), right.toInt()).toByte()
inline fun Math.floorMod(left: Short, right: Short) = floorMod(left.toInt(), right.toInt()).toShort()

fun Math.floorMod(left: Float, right: Float): Float {
    var mod = left % right

    val modLessZero = mod < 0f
    val rightLessZero = right < 0f

    if (modLessZero xor rightLessZero && mod != 0f) {
        mod += right
    }

    return mod
}

fun Math.floorMod(left: Double, right: Double): Double {
    var mod = left % right

    val modLessZero = mod < 0.0
    val rightLessZero = right < 0.0

    if (modLessZero xor rightLessZero && mod != 0.0) {
        mod += right
    }

    return mod
}
