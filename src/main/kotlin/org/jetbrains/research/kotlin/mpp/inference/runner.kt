package org.jetbrains.research.kotlin.mpp.inference

import java.io.File

fun main() {
    val model = ModelProto.ADAPTER.decode(File("/Users/username/Desktop/t.onnx").readBytes())
    println()
}
