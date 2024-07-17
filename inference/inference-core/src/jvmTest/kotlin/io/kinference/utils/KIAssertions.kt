package io.kinference.utils

import io.kinference.TestLoggerFactory
import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.arrays.*
import io.kinference.protobuf.FLOAT_TENSOR_TYPES
import io.kinference.protobuf.message.TensorProto
import kotlin.test.assertEquals

object KIAssertions {
    private val logger = TestLoggerFactory.create("io.kinference.utils.KIAssertions")

    fun assertEquals(expected: KITensor, actual: KITensor, delta: Double) {
        assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.name} do not match")
        ArrayAssertions.assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray()) { "Shapes of tensors ${expected.name} do not match" }

        val typeInfo = expected.info
        when (typeInfo.type) {
            in FLOAT_TENSOR_TYPES -> {
                val expectedArray = (expected.data as FloatNDArray).array.blocks
                val actualArray = (actual.data as FloatNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array.blocks
                val actualArray = (actual.data as DoubleNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array.blocks
                val actualArray = (actual.data as LongNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array.blocks
                val actualArray = (actual.data as IntNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.INT16 -> {
                val expectedArray = (expected.data as ShortNDArray).array.blocks
                val actualArray = (actual.data as ShortNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.INT8 -> {
                val expectedArray = (expected.data as ByteNDArray).array.blocks
                val actualArray = (actual.data as ByteNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.blocks
                val actualArray = (actual.data as BooleanNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            TensorProto.DataType.UINT8 -> {
                val expectedArray = (expected.data as UByteNDArray).array.blocks
                val actualArray = (actual.data as UByteNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }

            TensorProto.DataType.UINT16 -> {
                val expectedArray = (expected.data as UShortNDArray).array.blocks
                val actualArray = (actual.data as UShortNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }

            TensorProto.DataType.UINT32 -> {
                val expectedArray = (expected.data as UIntNDArray).array.blocks
                val actualArray = (actual.data as UIntNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }

            TensorProto.DataType.UINT64 -> {
                val expectedArray = (expected.data as ULongNDArray).array.blocks
                val actualArray = (actual.data as ULongNDArray).array.blocks

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }

            TensorProto.DataType.STRING -> {
                val expectedArray = (expected.data as StringNDArray).array
                val actualArray = (actual.data as StringNDArray).array

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray) { "Tensors ${expected.name} do not match" }
            }

            else -> assertEquals(expected, actual, "Tensors ${expected.name} do not match")
        }
    }

    fun assertEquals(expected: KIONNXMap, actual: KIONNXMap, delta: Double) {
        assertEquals(expected.keyType, actual.keyType, "Map key types should match")
        assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

        for (entry in expected.data.entries) {
            assertEquals(entry.value, actual.data[entry.key]!!, delta)
        }
    }

    fun assertEquals(expected: KIONNXSequence, actual: KIONNXSequence, delta: Double) {
        assertEquals(expected.length, actual.length, "Sequence lengths do not match")

        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], delta)
        }
    }

    fun assertEquals(expected: KIONNXData<*>, actual: KIONNXData<*>, delta: Double) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as KITensor, actual as KITensor, delta)
            ONNXDataType.ONNX_MAP -> assertEquals(expected as KIONNXMap, actual as KIONNXMap, delta)
            ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as KIONNXSequence, actual as KIONNXSequence, delta)
        }
    }
}
