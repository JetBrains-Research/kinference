package io.kinference.utils

import io.kinference.ndarray.arrays.tiled.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object ArrayAssertions {
    fun assertArrayEquals(left: FloatTiledArray, right: FloatTiledArray, diff: (Float, Float) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: FloatArray, right: FloatArray, diff: (Float, Float) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: DoubleTiledArray, right: DoubleTiledArray, diff: (Double, Double) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: DoubleArray, right: DoubleArray, diff: (Double, Double) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: LongTiledArray, right: LongTiledArray, diff: (Long, Long) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: LongArray, right: LongArray, diff: (Long, Long) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: IntTiledArray, right: IntTiledArray, diff: (Int, Int) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: IntArray, right: IntArray, diff: (Int, Int) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UIntArray, right: UIntArray, diff: (UInt, UInt) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UByteTiledArray, right: UByteTiledArray, diff: (UByte, UByte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UByteArray, right: UByteArray, diff: (UByte, UByte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: ShortArray, right: ShortArray, diff: (Short, Short) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: ByteArray, right: ByteArray, diff: (Byte, Byte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun <T : Comparable<T>> assertArrayEquals(left: Array<T>, right: Array<T>, diff: (T, T) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertTrue(diff(l, r) <= delta, message)
        }
    }


    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: String = "") {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertEquals(l, r, message)
        }
    }
}
