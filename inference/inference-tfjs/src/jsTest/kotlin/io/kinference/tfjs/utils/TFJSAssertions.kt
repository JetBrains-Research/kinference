package io.kinference.tfjs.utils

import io.kinference.TestLoggerFactory
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.utils.ArrayAssertions
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

object TFJSAssertions {
    private val logger = TestLoggerFactory.create("Assertions")

    fun assertEquals(expected: TFJSTensor, actual: TFJSTensor, delta: Double, verboseErrors: Boolean = true) {
        assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.name} do not match")
        ArrayAssertions.assertArrayEquals(expected.data.shapeArray, actual.data.shapeArray, "Shapes are incorrect")


        when(expected.data.type) {
            DataType.FLOAT -> {
                val expectedArray = expected.data.dataFloat()
                val actualArray = actual.data.dataFloat()

                if (verboseErrors) {
                    logger.info { "Errors for ${expected.name}:" }
                    ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
                } else {
                    ArrayAssertions.assertArrayEquals(
                        left = expectedArray,
                        right = actualArray,
                        diff = { l, r -> abs(l - r).toDouble() },
                        delta = delta,
                        message = "Tensor ${expected.name.orEmpty()} does not match")
                }
            }

            DataType.INT -> {
                val expectedArray = expected.data.dataInt()
                val actualArray = actual.data.dataInt()

                if (verboseErrors) {
                    logger.info { "Errors for ${expected.name}:" }
                    ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
                } else {
                    ArrayAssertions.assertArrayEquals(
                        left = expectedArray,
                        right = actualArray,
                        diff = { l, r -> abs(l - r).toDouble() },
                        delta = delta,
                        message = "Tensor ${expected.name.orEmpty()} does not match")
                }
            }

            DataType.BOOLEAN -> {
                val expectedArray = expected.data.dataBool()
                val actualArray = actual.data.dataBool()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.name} does not match")
            }

            else -> error("Unsupported data type of ${expected.name} tensor")
        }
    }

    fun assertEquals(expected: TFJSMap, actual: TFJSMap, delta: Double) {
        assertEquals(expected.keyType, actual.keyType, "Map key types should match")
        assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

        for (entry in expected.data.entries) {
            assertEquals(entry.value, actual.data[entry.key]!!, delta, verboseErrors = false)
        }
    }

    fun assertEquals(expected: TFJSSequence, actual: TFJSSequence, delta: Double) {
        assertEquals(expected.length, actual.length, "Sequence lengths do not match")

        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], delta, verboseErrors = false)
        }
    }

    fun assertEquals(expected: TFJSData<*>, actual: TFJSData<*>, delta: Double, verboseErrors: Boolean = true) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as TFJSTensor, actual as TFJSTensor, delta, verboseErrors)
            ONNXDataType.ONNX_MAP -> assertEquals(expected as TFJSMap, actual as TFJSMap, delta)
            ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as TFJSSequence, actual as TFJSSequence, delta)
        }
    }
}
