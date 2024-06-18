package io.kinference.tfjs.utils

import io.kinference.data.ONNXDataType
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.utils.*

object TFJSErrors {
    fun calculateErrors(expected: TFJSTensor, actual: TFJSTensor): Errors.ErrorsData {
        return when (expected.data.type) {
            DataType.FLOAT -> {
                val expectedArray = expected.data.dataFloat()
                val actualArray = actual.data.dataFloat()

                Errors.computeErrors(expectedArray, actualArray)
            }

            DataType.INT -> {
                val expectedArray = expected.data.dataInt()
                val actualArray = actual.data.dataInt()

                Errors.computeErrors(expectedArray, actualArray)
            }

            DataType.BOOLEAN -> {
                val expectedArray = expected.data.dataBool()
                val actualArray = actual.data.dataBool()

                Errors.computeErrors(expectedArray.toBooleanArray(), actualArray.toBooleanArray())
            }

            else -> error("Unsupported tensor data type ${expected.info.type}")
        }
    }

    fun calculateErrors(expected: TFJSSequence, actual: TFJSSequence): List<Errors.ErrorsData> {
        val result = mutableListOf<Errors.ErrorsData>()
        for (idx in expected.data.indices) {
            result.addAll(calculateErrors(expected.data[idx], actual.data[idx]))
        }
        return result
    }

    fun calculateErrors(expected: TFJSMap, actual: TFJSMap): List<Errors.ErrorsData> {
        val result = mutableListOf<Errors.ErrorsData>()
        for (entry in expected.data.entries) {
            result.addAll(calculateErrors(entry.value, actual.data[entry.key]!!))
        }

        return result
    }

    fun calculateErrors(expected: TFJSData<*>, actual: TFJSData<*>): List<Errors.ErrorsData> {
        return when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> listOf(calculateErrors(expected as TFJSTensor, actual as TFJSTensor))
            ONNXDataType.ONNX_MAP -> calculateErrors(expected as TFJSMap, actual as TFJSMap)
            ONNXDataType.ONNX_SEQUENCE -> calculateErrors(expected as TFJSSequence, actual as TFJSSequence)
        }
    }
}
