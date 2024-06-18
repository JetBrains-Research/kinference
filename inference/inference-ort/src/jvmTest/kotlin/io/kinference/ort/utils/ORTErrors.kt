package io.kinference.ort.utils

import ai.onnxruntime.OnnxJavaType
import io.kinference.data.ONNXDataType
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.*
import io.kinference.ort.ORTData


object ORTErrors {
    fun calculateErrors(expected: ORTTensor, actual: ORTTensor): Errors.ErrorsData {
        require(expected.type == actual.type)

        return when(expected.data.info.type) {
            in FLOATS -> {
                val expectedArray = expected.toFloatArray()
                val actualArray = actual.toFloatArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.DOUBLE -> {
                val expectedArray = expected.toDoubleArray()
                val actualArray = actual.toDoubleArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.INT8 -> {
                val expectedArray = expected.toByteArray()
                val actualArray = actual.toByteArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.INT16 -> {
                val expectedArray = expected.toShortArray()
                val actualArray = actual.toShortArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.INT32 -> {
                val expectedArray = expected.toIntArray()
                val actualArray = actual.toIntArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.INT64 -> {
                val expectedArray = expected.toLongArray()
                val actualArray = actual.toLongArray()

                Errors.computeErrors(expectedArray, actualArray)
            }

            OnnxJavaType.UINT8 -> {
                val expectedArray = expected.toUByteArray()
                val actualArray = actual.toUByteArray()

                Errors.computeErrors(expectedArray, actualArray)
            }
            else -> error("Unsupported data type: ${actual.type}")
        }
    }

    fun calculateErrors(expected: ORTData<*>, actual: ORTData<*>): List<Errors.ErrorsData> {
        require(expected.type == ONNXDataType.ONNX_TENSOR && actual.type == ONNXDataType.ONNX_TENSOR)
        return listOf(calculateErrors(expected as ORTTensor, actual as ORTTensor))
    }
}
