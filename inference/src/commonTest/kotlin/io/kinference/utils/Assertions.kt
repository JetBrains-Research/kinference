package io.kinference.utils

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.map.ONNXMap
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.*
import io.kinference.onnx.TensorProto
import io.kinference.types.ValueTypeInfo
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object Assertions {
    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: Tensor, actual: Tensor, delta: Double) {
        assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray(), "Shapes are incorrect")

        val typeInfo = expected.info.typeInfo as ValueTypeInfo.TensorTypeInfo
        when (typeInfo.type) {
            TensorProto.DataType.FLOAT -> {
                val expectedArray = (expected.data as FloatNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as FloatNDArray).array.toArray().toTypedArray()
                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as DoubleNDArray).array.toArray().toTypedArray()
                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r) }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as LongNDArray).array.toArray().toTypedArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as IntNDArray).array.toArray().toTypedArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as BooleanNDArray).array.toArray().toTypedArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.UINT8 -> {
                val actualArray = (actual.data as UByteNDArray).array.toArray()
                ((expected.data as UByteNDArray).array.toArray()).forEachIndexed { index, value ->
                    assertEquals(value, actualArray[index], "Tensor ${expected.info.name} does not match")
                }
            }
            else -> assertEquals(expected, actual, "Tensor ${expected.info.name} does not match")
        }
    }

    fun <T : Number> assertArrayEquals(left: Array<T>, right: Array<T>, diff: (T, T) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: String) {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertEquals(l, r, message)
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
