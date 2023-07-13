package io.kinference.utils

import kotlin.test.assertTrue

object Assertions {
    fun <T: Comparable<T>> assertLessOrEquals(expected: T, actual: T, message: String) {
        assertTrue(actual <= expected, message)
    }
}
