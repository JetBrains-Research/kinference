package io.kinference.ndarray.math

import kotlin.math.*

actual object FastMath {
    actual inline fun exp(value: Double) = kotlin.math.exp(value)

    actual inline fun copySign(value: Double, sign: Double) = value.withSign(sign)
    actual inline fun copySign(value: Float, sign: Float) = value.withSign(sign)
    actual inline fun pow(value: Double, e: Long) = value.pow(e.toDouble())
}
