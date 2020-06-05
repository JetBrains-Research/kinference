package org.jetbrains.research.kotlin.mpp.inference

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpace
import org.jetbrains.research.kotlin.mpp.inference.space.toIntArray
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.junit.jupiter.api.Assertions.*
import scientifik.kmath.structures.*
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow

object Utils {
    private val delta = (10.0).pow(-6)

    fun getTensor(path: File) : Tensor<*> {
        val tensorProto = TensorProto.ADAPTER.decode(path.readBytes())
        return when (DataType.fromValue(tensorProto.data_type!!) ?: 0) {
            DataType.FLOAT -> getTensorFloat(tensorProto)
            else -> throw UnsupportedOperationException()
        }
    }

    private fun getTensorFloat(tensorProto : TensorProto) : Tensor<Float> {
        val rawFloatData = tensorProto.raw_data!!.toByteArray()
        val chunkedRawFloatData = rawFloatData.asIterable().chunked(4)
        val floatData = chunkedRawFloatData.map { ByteBuffer.wrap(it.reversed().toByteArray()).float }.asBuffer()
        val structure = BufferNDStructure(SpaceStrides(tensorProto.dims.toIntArray()), floatData)
        return Tensor(tensorProto.name, structure, DataType.FLOAT, resolveSpace(tensorProto.dims))
    }

    fun assertTensors(expected: Tensor<*>, actual: Tensor<*>) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)
        @Suppress("UNCHECKED_CAST")
        when (expected.type) {
            DataType.FLOAT -> {
                expected as Tensor<Float>
                actual as Tensor<Float>
                expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                    assertEquals(value, actual.data.buffer[index], delta.toFloat(), "Tensor ${expected.name} does not match")
                }
            }

            DataType.DOUBLE -> {
                expected as Tensor<Double>
                actual as Tensor<Double>
                expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                    assertEquals(value, actual.data.buffer[index], delta, "Tensor ${expected.name} does not match")
                }
            }

            else -> assertEquals(expected, actual, "Tensor ${expected.name} does not match")
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun operatorTestHelper(folderName: String): List<Pair<List<Tensor<Number>>, List<Tensor<Number>>>> {
        val path = javaClass.getResource(folderName).path
        val model = Model.load(path + "model.onnx")

        return File(path).list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file ->  "input" in file.name }
            val outputFiles = File("$path/$it").walk().filter { file -> "output" in file.name }

            val inputTensors = inputFiles.map { getTensor(it) }.toList() as List<Tensor<Number>>
            val expectedOutputTensors = outputFiles.map { getTensor(it) }.toList() as List<Tensor<Number>>
            val actualOutputTensors = model.predict(inputTensors) as List<Tensor<Number>>
            Pair(expectedOutputTensors, actualOutputTensors)
        }
    }
}
