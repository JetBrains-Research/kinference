package io.kinference.ort.utils

import ai.onnxruntime.*
import io.kinference.data.ONNXDataType
import io.kinference.ort.ORTData
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.ArrayAssertions

object ORTAssertions {
    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        require(expected.type == ONNXDataType.ONNX_TENSOR && actual.type == ONNXDataType.ONNX_TENSOR)
        assertTensorEquals(expected as ORTTensor, actual as ORTTensor, delta)
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertTensorEquals(expected: ORTTensor, actual: ORTTensor, delta: Double) {
        kotlin.test.assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
        ArrayAssertions.assertArrayEquals(expected.shape.toTypedArray(), actual.shape.toTypedArray(), "Shapes are incorrect")


        when (expected.data.info.type) {
            OnnxJavaType.FLOAT ->  {
                val expectedArray = expected.data.floatBuffer.array()
                val actualArray = actual.data.floatBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            OnnxJavaType.DOUBLE -> {
                val expectedArray = expected.data.doubleBuffer.array()
                val actualArray = actual.data.doubleBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            OnnxJavaType.INT32 -> {
                val expectedArray = expected.data.intBuffer.array()
                val actualArray = actual.data.intBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            OnnxJavaType.INT64 -> {
                val expectedArray = expected.data.longBuffer.array()
                val actualArray = actual.data.longBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            OnnxJavaType.INT16 -> {
                val expectedArray = expected.data.shortBuffer.array()
                val actualArray = actual.data.shortBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            OnnxJavaType.INT8, OnnxJavaType.BOOL -> {
                val expectedArray = expected.data.byteBuffer.array()
                val actualArray = actual.data.byteBuffer.array()

                ArrayAssertions.assertEquals(expectedArray, actualArray, delta, actual.name.orEmpty())
            }
            else -> error("Unsupported data type: ${expected.data.info.type}")
        }
    }
}

