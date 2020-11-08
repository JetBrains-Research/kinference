package io.kinference

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.model.Model
import io.kinference.ndarray.arrays.*
import io.kinference.onnx.TensorProto
import io.kinference.onnx.TensorProto.DataType
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import java.io.File
import kotlin.math.pow

object Utils {
    private val delta = (10.0).pow(-3)

    fun getTensor(file: File): Tensor = getTensor(file.readBytes())

    fun getTensor(byteArray: ByteArray): Tensor = getTensor(TensorProto.ADAPTER.decode(byteArray))

    fun getTensor(tensorProto: TensorProto) = Tensor.create(tensorProto)

    fun assertTensors(expected: Tensor, actual: Tensor, delta: Double) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)

        @Suppress("UNCHECKED_CAST")
        when (expected.info.type) {
            DataType.FLOAT -> {
                val expectedArray = (expected.data as FloatNDArray).array.toArray()
                val actualArray = (actual.data as FloatNDArray).array.toArray()
                assertArrayEquals(expectedArray, actualArray, delta.toFloat(), "Tensor ${expected.info.name} does not match")
            }

            DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array.toArray()
                val actualArray = (actual.data as DoubleNDArray).array.toArray()
                assertArrayEquals(expectedArray, actualArray, delta, "Tensor ${expected.info.name} does not match")
            }

            DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array.toArray()
                val actualArray = (actual.data as LongNDArray).array.toArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }

            DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array.toArray()
                val actualArray = (actual.data as IntNDArray).array.toArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }

            DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.toArray()
                val actualArray = (actual.data as BooleanNDArray).array.toArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }

            DataType.UINT8 -> {
                val actualArray = (actual.data as UByteNDArray).array.toArray()
                ((expected.data as UByteNDArray).array.toArray()).forEachIndexed { index, value ->
                    assertEquals(value, actualArray[index], "Tensor ${expected.info.name} does not match")
                }
            }

            else -> assertEquals(expected, actual, "Tensor ${expected.info.name} does not match")
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun operatorTestHelper(folderName: String): List<Pair<List<ONNXData>, List<ONNXData>>> {
        val path = javaClass.getResource(folderName).path
        val model = Model.load(path + "model.onnx")

        return File(path).list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
            val outputFiles = File("$path/$it").walk().filter { file -> "output" in file.name }

            val inputTensors = inputFiles.map { getTensor(it) }.toList()
            val expectedOutputTensors = outputFiles.map { getTensor(it) }.toList()
            val actualOutputTensors = model.predict(inputTensors)
            Pair(expectedOutputTensors, actualOutputTensors)
        }
    }

    fun tensorTestRunner(testDir: String, delta: Double = this.delta) {
        val dataSets = operatorTestHelper(testDir)
        for (dataSet in dataSets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

            for (expectedOutputTensor in expectedOutputTensors) {
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
                assertTensors(expectedOutputTensor as Tensor, actualOutputTensor as Tensor, delta)
            }
        }
    }
}
