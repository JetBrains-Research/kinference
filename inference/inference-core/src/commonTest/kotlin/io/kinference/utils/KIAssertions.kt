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
import io.kinference.utils.ArrayAssertions.assertArrayEquals
import kotlin.test.assertEquals

object KIAssertions {
    private val logger = TestLoggerFactory.create("io.kinference.utils.KIAssertions")

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: KITensor, actual: KITensor, delta: Double) {
        assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.name} do not match")
        assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray(), "Shapes are incorrect")

        val typeInfo = expected.info
        logger.info { "Errors for ${expected.name}:" }
        when (typeInfo.type) {
            in FLOAT_TENSOR_TYPES -> {
                val expectedArray = (expected.data as FloatNDArray).array
                val actualArray = (actual.data as FloatNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
            TensorProto.DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array
                val actualArray = (actual.data as DoubleNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
            TensorProto.DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array
                val actualArray = (actual.data as LongNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array
                val actualArray = (actual.data as IntNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }
            TensorProto.DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as BooleanNDArray).array.toArray().toTypedArray()

                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.name} does not match")
            }
            TensorProto.DataType.UINT8 -> {
                val expectedArray = (expected.data as UByteNDArray).array
                val actualArray = (actual.data as UByteNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }

            TensorProto.DataType.UINT16 -> {
                val expectedArray = (expected.data as UShortNDArray).array
                val actualArray = (actual.data as UShortNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }

            TensorProto.DataType.UINT32 -> {
                val expectedArray = (expected.data as UIntNDArray).array
                val actualArray = (actual.data as UIntNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }

            TensorProto.DataType.UINT64 -> {
                val expectedArray = (expected.data as ULongNDArray).array
                val actualArray = (actual.data as ULongNDArray).array

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, expected.name.orEmpty())
            }

            else -> assertEquals(expected, actual, "Tensor ${expected.name} does not match")
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
