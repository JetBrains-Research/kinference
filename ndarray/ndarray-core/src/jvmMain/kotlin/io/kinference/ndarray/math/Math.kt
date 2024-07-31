package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType
import java.lang.Math

object Math {
    fun floorMod(x: Int, y: Int): Int = Math.floorMod(x, y)
    fun floorMod(x: Long, y: Long): Long = Math.floorMod(x, y)
    internal inline fun floorMod(x: PrimitiveType, y: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
}

//internal inline fun Math.floorMod(x: PrimitiveType, y: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
