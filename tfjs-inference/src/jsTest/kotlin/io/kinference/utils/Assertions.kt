package io.kinference.utils

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.map.ONNXMap
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.ndarray.logger
import org.khronos.webgl.*
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object Assertions {
    val logger = logger("Errors")

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: Tensor, actual: Tensor, delta: Double) {
        assertEquals(expected.data.dtype, actual.data.dtype, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape, "Shapes are incorrect")


        when(expected.data.dtype) {
            "float32" -> {
                val expectedArray = expected.data.dataSync() as Float32Array
                val actualArray = actual.data.dataSync() as Float32Array
                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            "int32" -> {
                val expectedArray = expected.data.dataSync() as Int32Array
                val actualArray = actual.data.dataSync() as Int32Array

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
        }
    }

    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: String) {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertEquals(l, r, message)
        }
    }

    fun assertArrayEquals(left: Float32Array, right: Float32Array, diff: (Float, Float) -> Double, delta: Double, message: String) {
        assertEquals(left.length, right.length, message)
        for (i in 0 until left.length) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: Int32Array, right: Int32Array, diff: (Int, Int) -> Double, delta: Double, message: String) {
        assertEquals(left.length, right.length, message)
        for (i in 0 until left.length) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertEquals(expected: ONNXMap, actual: ONNXMap, delta: Double) {
        assertEquals(expected.keyType, actual.keyType, "Map key types should match")
        assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

        for (entry in expected.data.entries) {
            assertEquals(entry.value, actual.data[entry.key]!!, delta)
        }
    }

    fun assertEquals(expected: ONNXSequence, actual: ONNXSequence, delta: Double) {
        assertEquals(expected.length, actual.length, "Sequence lengths do not match")

        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], delta)
        }
    }

    fun assertEquals(expected: ONNXData, actual: ONNXData, delta: Double) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as Tensor, actual as Tensor, delta)
            ONNXDataType.ONNX_MAP -> assertEquals(expected as ONNXMap, actual as ONNXMap, delta)
            ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as ONNXSequence, actual as ONNXSequence, delta)
        }
    }
}
