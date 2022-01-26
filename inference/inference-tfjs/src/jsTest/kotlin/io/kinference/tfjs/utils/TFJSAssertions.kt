package io.kinference.tfjs.utils

import io.kinference.TestLoggerFactory
import io.kinference.data.ONNXDataType
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.externals.extensions.dataFloat
import io.kinference.tfjs.externals.extensions.dataInt
import io.kinference.utils.ArrayAssertions
import kotlin.test.assertEquals

object TFJSAssertions {
    val logger = TestLoggerFactory.create("Assertions")

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: TFJSTensor, actual: TFJSTensor, delta: Double) {
        assertEquals(expected.data.dtype, actual.data.dtype, "Types of tensors ${expected.name} do not match")
        ArrayAssertions.assertArrayEquals(expected.data.shape, actual.data.shape, "Shapes are incorrect")


        logger.info { "Errors for ${expected.name}:" }
        when(expected.data.dtype) {
            "float32" -> {
                val expectedArray = expected.data.dataFloat()
                val actualArray = actual.data.dataFloat()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
            "int32" -> {
                val expectedArray = expected.data.dataInt()
                val actualArray = actual.data.dataInt()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
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
