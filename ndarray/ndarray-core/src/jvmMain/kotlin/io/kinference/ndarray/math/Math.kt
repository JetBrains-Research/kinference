package io.kinference.ndarray.math

import java.lang.Math

actual object Math {
    actual inline fun floorMod(x: Int, y: Int): Int = Math.floorMod(x, y)

    actual inline fun floorMod(x: Long, y: Long): Long = Math.floorMod(x, y)
}
