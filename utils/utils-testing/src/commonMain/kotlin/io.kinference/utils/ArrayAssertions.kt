package io.kinference.utils

import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object ArrayAssertions {
    fun <T : Number> assertArrayEquals(left: Array<T>, right: Array<T>, delta: Double, message: () -> String) {
        val message = message()

        assertEquals(left.size, right.size, message)
        for (idx in left.indices) {
            val l = left[idx].toDouble()
            val r = right[idx].toDouble()

            assertTrue(abs(l - r) <= delta, message)
        }
    }


    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: () -> String) {
        val message = message()

        assertEquals(left.size, right.size, message)
        for (idx in left.indices) {
            val l = left[idx]
            val r = right[idx]

            assertEquals(l, r, message)
        }
    }
}
