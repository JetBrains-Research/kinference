package io.kinference.ndarray.math

import org.apache.commons.math4.core.jdkmath.AccurateMath

actual object FastMath {
    actual inline fun exp(value: Double) = AccurateMath.exp(value)

    actual inline fun copySign(value: Double, sign: Double) = AccurateMath.copySign(value, sign)
    actual inline fun copySign(value: Float, sign: Float) = AccurateMath.copySign(value, sign)
    actual inline fun pow(value: Double, e: Long) = AccurateMath.pow(value, e)
}
