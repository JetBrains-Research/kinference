@file:GeneratePrimitives(DataType.ALL)
package io.kinference.utils

import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.math.abs


@MakePublic
internal fun ArrayAssertions.assertArrayEquals(left: PrimitiveArray, right: PrimitiveArray, delta: Double, message: () -> String) {
    val message = message()

    assertEquals(left.size, right.size, message)

    for (idx in 0 until left.size) {
        val l = left[idx].toDouble()
        val r = right[idx].toDouble()

        assertTrue(abs(l - r) <= delta, message)
    }
}

@MakePublic
internal fun ArrayAssertions.assertArrayEquals(leftBlocks: Array<PrimitiveArray>, rightBlocks: Array<PrimitiveArray>, delta: Double, message: () -> String) {
    val message = message()

    assertEquals(leftBlocks.size, rightBlocks.size, message)

    for (blockIdx in leftBlocks.indices) {
        val lBlock = leftBlocks[blockIdx]
        val rBlock = rightBlocks[blockIdx]

        assertEquals(lBlock.size, rBlock.size, message)

        for (idx in lBlock.indices) {
            val l = lBlock[idx].toDouble()
            val r = rBlock[idx].toDouble()

            assertTrue(abs(l - r) <= delta, message)
        }
    }
}
