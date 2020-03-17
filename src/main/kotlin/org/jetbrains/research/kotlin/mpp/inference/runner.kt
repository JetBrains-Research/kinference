package org.jetbrains.research.kotlin.mpp.inference

import org.jetbrains.research.kotlin.mpp.inference.model.Model
import scientifik.kmath.structures.asBuffer

fun main() {
    val tokens = loadMapping("/Users/username/DeepBugsPlugin/js-plugin/src/main/models/tokenToVector.cbor")
    val types = loadMapping("/Users/username/DeepBugsPlugin/js-plugin/src/main/models/typeToVector.cbor")
    val name = tokens["ID:setSize"]!!
    val arg1 = tokens["ID:y"]!!
    val arg2 = tokens["ID:x"]!!
    val base = FloatArray(200) { 0.0f }
    val param1 = tokens["ID:x"]!!
    val param2 = tokens["ID:y"]!!
    val type = types["unknown"]!!
    val t = name + arg1 + arg2 + base + type + type + param1 + param2

    val model = Model.load("/Users/username/PycharmProjects/model-converter/swappedArgsDetectionModel.onnx")
    val res = model.predict(t.toList())
    println(res.asBuffer().list)
}
