package io.kinference.ndarray.math

actual object Math {
    actual fun floorMod(x: Int, y: Int): Int {
        var mod = x % y
        if ((x or y) < 0 && mod != 0) {
            mod += y
        }

        return mod
    }

    actual fun floorMod(x: Long, y: Long): Long {
        var mod = x % y
        if ((x or y) < 0 && mod != 0L) {
            mod += y
        }

        return mod
    }
}
