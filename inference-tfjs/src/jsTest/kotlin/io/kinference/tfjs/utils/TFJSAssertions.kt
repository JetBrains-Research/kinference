package io.kinference.tfjs.utils

import io.kinference.data.ONNXDataType
import io.kinference.ndarray.logger
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import org.khronos.webgl.*
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object TFJSAssertions {
    val logger = logger("Errors")

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: TFJSTensor, actual: TFJSTensor, delta: Double) {
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

    fun assertEquals(expected: TFJSMap, actual: TFJSMap, delta: Double) {
        assertEquals(expected.keyType, actual.keyType, "Map key types should match")
        assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

        for (entry in expected.data.entries) {
            assertEquals(entry.value, actual.data[entry.key]!!, delta)
        }
    }

    fun assertEquals(expected: TFJSSequence, actual: TFJSSequence, delta: Double) {
        assertEquals(expected.length, actual.length, "Sequence lengths do not match")

        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], delta)
        }
    }

    fun assertEquals(expected: TFJSData<*>, actual: TFJSData<*>, delta: Double) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as TFJSTensor, actual as TFJSTensor, delta)
            ONNXDataType.ONNX_MAP -> assertEquals(expected as TFJSMap, actual as TFJSMap, delta)
            ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as TFJSSequence, actual as TFJSSequence, delta)
        }
    }
}
