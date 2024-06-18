package io.kinference.ort.utils

import ai.onnxruntime.*
import io.kinference.data.ONNXDataType
import io.kinference.ort.ORTData
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.ArrayAssertions
import io.kinference.utils.assertArrayEquals
import kotlin.test.assertEquals

object ORTAssertions {
    fun assertEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        require(expected.type == ONNXDataType.ONNX_TENSOR && actual.type == ONNXDataType.ONNX_TENSOR)
        assertTensorEquals(expected as ORTTensor, actual as ORTTensor, delta)
    }

    fun assertTensorEquals(expected: ORTTensor, actual: ORTTensor, delta: Double) {
        assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
        ArrayAssertions.assertArrayEquals(expected.shape.toTypedArray(), actual.shape.toTypedArray()) { "Shapes of tensors ${expected.name} do not match" }


        when (expected.data.info.type) {
            OnnxJavaType.FLOAT ->  {
                val expectedArray = expected.data.floatBuffer.array()
                val actualArray = actual.data.floatBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            OnnxJavaType.DOUBLE -> {
                val expectedArray = expected.data.doubleBuffer.array()
                val actualArray = actual.data.doubleBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            OnnxJavaType.INT32 -> {
                val expectedArray = expected.data.intBuffer.array()
                val actualArray = actual.data.intBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            OnnxJavaType.INT64 -> {
                val expectedArray = expected.data.longBuffer.array()
                val actualArray = actual.data.longBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            OnnxJavaType.INT16 -> {
                val expectedArray = expected.data.shortBuffer.array()
                val actualArray = actual.data.shortBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            OnnxJavaType.INT8, OnnxJavaType.BOOL -> {
                val expectedArray = expected.data.byteBuffer.array()
                val actualArray = actual.data.byteBuffer.array()

                ArrayAssertions.assertArrayEquals(expectedArray, actualArray, delta) { "Tensors ${expected.name} do not match" }
            }
            else -> error("Unsupported data type: ${expected.data.info.type}")
        }
    }
}

