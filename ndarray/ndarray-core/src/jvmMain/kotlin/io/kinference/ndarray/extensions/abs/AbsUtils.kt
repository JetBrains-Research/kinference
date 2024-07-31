package io.kinference.ndarray.extensions.abs

import kotlin.math.abs

internal fun abs(x: Short) = abs(x.toInt()).toShort()
internal fun abs(x: Byte) = abs(x.toInt()).toByte()
