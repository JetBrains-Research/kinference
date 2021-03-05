package io.kinference.models.license_detector

import io.kinference.data.ONNXData
import io.kinference.data.map.ONNXMap
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.protobuf.message.TensorProto
import io.kinference.runners.TestRunner
import io.kinference.types.*
import io.kinference.utils.DataLoader
import kotlinx.serialization.DeserializationStrategy
import kotlinx.serialization.builtins.*
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.io.File

class LicenseDetectorTest {
    @Test
    @Tag("heavy")
    fun `test detector`() {
        TestRunner.runFromS3(TEST_PATH, PREFIX, ::licenseDetectorTestRunner)
    }

    companion object {
        private val json = Json.Default
        private val deserializer = ListSerializer(ListSerializer(MapSerializer(Long.serializer(), Float.serializer())))

        private inline fun <reified T> parse(serializer: DeserializationStrategy<T>, value: String): T {
            return json.decodeFromString(serializer, value)
        }

        fun licenseDetectorTestRunner(path: File): List<TestRunner.ONNXTestData> {
            val model = Model.load(File(path, "model.onnx").absolutePath)

            return path.list()!!.filter { "test" in it }.map {
                val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
                val outputLabels = File("$path/$it").walk().find { file -> "output_labels.pb" == file.name }!!
                val outputScores = File("$path/$it").walk().find { file -> "output_scores.json" == file.name }!!

                val inputTensors = inputFiles.map { model.graph.prepareInput(TensorProto.decode(it.readBytes())) }.toList()
                val expectedLabels = DataLoader.getTensor(outputLabels)
                val expectedScores = loadJsonONNXSequence(outputScores)
                val actual = model.predict(inputTensors.toList())

                TestRunner.ONNXTestData(actual, listOf(expectedLabels, expectedScores))
            }
        }

        private fun loadJsonONNXSequence(file: File): ONNXSequence {
            val mapList = parse(deserializer, file.readText())[0]
            val info = ValueTypeInfo.TensorTypeInfo(TensorShape.empty(), TensorProto.DataType.FLOAT)
            val elementType = ValueTypeInfo.MapTypeInfo(TensorProto.DataType.INT64, info)

            val valueInfo = ValueTypeInfo.SequenceTypeInfo(elementType)
            return ONNXSequence(ValueInfo(valueInfo, "probabilities"), mapList.size) {
                val map = mapList[it].mapValues { FloatNDArray.scalar(it.value).asTensor() }
                ONNXMap(map as Map<Any, ONNXData>, ValueInfo(elementType))
            }
        }

        const val TEST_PATH = "/catboost/license_detector/"
        const val PREFIX = "tests/catboost/license_detector/"
    }
}
