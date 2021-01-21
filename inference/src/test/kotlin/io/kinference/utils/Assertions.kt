package io.kinference.utils

import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.*
import io.kinference.onnx.TensorProto
import io.kinference.types.ValueTypeInfo
import org.junit.jupiter.api.Assertions

object Assertions {
    fun assertEquals(expected: Tensor, actual: Tensor, delta: Double) {
        Assertions.assertEquals(expected.type, actual.type, "Types of tensors ${expected.info.name} do not match")
        Assertions.assertArrayEquals(expected.data.shape, actual.data.shape)

        val typeInfo = expected.info.typeInfo as ValueTypeInfo.TensorTypeInfo
        when (typeInfo.type) {
            TensorProto.DataType.FLOAT -> {
                val expectedArray = (expected.data as FloatNDArray).array.toArray()
                val actualArray = (actual.data as FloatNDArray).array.toArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, delta.toFloat(), "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array.toArray()
                val actualArray = (actual.data as DoubleNDArray).array.toArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array.toArray()
                val actualArray = (actual.data as LongNDArray).array.toArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array.toArray()
                val actualArray = (actual.data as IntNDArray).array.toArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.toArray()
                val actualArray = (actual.data as BooleanNDArray).array.toArray()
                Assertions.assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.UINT8 -> {
                val actualArray = (actual.data as UByteNDArray).array.toArray()
                ((expected.data as UByteNDArray).array.toArray()).forEachIndexed { index, value ->
                    Assertions.assertEquals(value, actualArray[index], "Tensor ${expected.info.name} does not match")
                }
            }
            else -> Assertions.assertEquals(expected, actual, "Tensor ${expected.info.name} does not match")
        }
    }

}
