package io.kinference.models.catboost

class IJLicenseDetectorTest {
//    @Test
//    @Tag("heavy")
//    fun `test detector`() {
//        TestRunner.runFromS3("catboost:license-detector:v1", ::licenseDetectorTestRunner)
//    }
//
//    companion object {
//        private val json = Json.Default
//        private val deserializer = ListSerializer(ListSerializer(MapSerializer(Long.serializer(), Float.serializer())))
//
//        private inline fun <reified T> parse(serializer: DeserializationStrategy<T>, value: String): T {
//            return json.decodeFromString(serializer, value)
//        }
//
//        fun licenseDetectorTestRunner(path: File): List<TestRunner.ONNXTestData> {
//            val model = Model.load(File(path, "model.onnx").absolutePath)
//
//            return path.list()!!.filter { "test" in it }.map {
//                val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
//                val outputLabels = File("$path/$it").walk().find { file -> "output_labels.pb" == file.name }!!
//                val outputScores = File("$path/$it").walk().find { file -> "output_scores.json" == file.name }!!
//
//                val inputTensors = inputFiles.map { model.graph.prepareInput(TensorProto.ADAPTER.decode(it.readBytes())) }.toList()
//                val expectedLabels = DataLoader.getTensor(outputLabels)
//                val expectedScores = loadJsonONNXSequence(outputScores)
//                val actual = model.predict(inputTensors)
//
//                TestRunner.ONNXTestData(actual, listOf(expectedLabels, expectedScores))
//            }
//        }
//
//        private fun loadJsonONNXSequence(file: File): ONNXSequence {
//            val mapList = parse(deserializer, file.readText())[0]
//            val info = ValueTypeInfo.TensorTypeInfo(TensorShape.empty(), TensorProto.DataType.FLOAT)
//            val elementType = ValueTypeInfo.MapTypeInfo(TensorProto.DataType.INT64, info)
//
//            val valueInfo = ValueTypeInfo.SequenceTypeInfo(elementType)
//            return ONNXSequence(ValueInfo(valueInfo, "probabilities"), mapList.size) {
//                val map = mapList[it].mapValues { FloatNDArray.scalar(it.value).asTensor() }
//                ONNXMap(map as Map<Any, ONNXData>, ValueInfo(elementType))
//            }
//        }
//    }
}
