package io.kinference.completion.generating

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class UtilsTest {
    @Test
    @Tag("heavy")
    fun testTopk1d() {
        val list = listOf(0.1, 0.2, 0.9, 0.3, 0.4, 0.0, 0.1, 0.5)
        assertEquals(topk1d(list, 3), listOf(2, 7, 4))
    }

    @Test
    @Tag("heavy")
    fun testTopk2d() {
        val list1 = listOf(0.1, 0.2, 0.9, 0.3, 0.4, 0.0, 0.1, 0.5)
        val list2 = listOf(0.6, 0.8, 0.2, 0.1, 0.4, 0.7, 0.3, 0.0)
        val list = listOf(list1, list2)
        assertEquals(topk2d(list, 3, dim = 1), listOf(listOf(2, 7, 4), listOf(1, 5, 0)))

        val column = listOf(
            listOf(0.9, 0.3),
            listOf(0.2, 0.2),
            listOf(0.1, 0.7),
            listOf(0.5, 0.4)
        )
        val target = listOf(
            listOf(0, 2),
            listOf(3, 3),
            listOf(1, 0)
        )
        assertEquals(topk2d(column, 3, dim = 0), target)
    }
}
