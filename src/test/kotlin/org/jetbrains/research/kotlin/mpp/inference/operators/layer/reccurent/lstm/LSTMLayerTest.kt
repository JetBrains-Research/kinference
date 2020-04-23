package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.assertTensors
import org.jetbrains.research.kotlin.mpp.inference.getTensor
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.junit.jupiter.api.Test
import java.io.File

class LSTMLayerTest {
    @Test
    fun `Default test`() {
        val path = javaClass.getResource("/lstm_defaults/").path
        val model = Model.load(path + "model.onnx")

        val inputFiles = File(path).walk().filter { "input" in it.name }
        val outputFiles = File(path).walk().filter { "output" in it.name }

        @Suppress("UNCHECKED_CAST")
        val inputTensors = inputFiles.map { getTensor(it) }.toList() as List<Tensor<Number>>
        @Suppress("UNCHECKED_CAST")
        val outputTensor = outputFiles.map { getTensor(it) }.toList() as List<Tensor<Number>>
        @Suppress("UNCHECKED_CAST")
        val actualOutput = model.predict(inputTensors) as Collection<Tensor<Number>>
        val mappedActualOutput = actualOutput.associateBy { it.name }
        outputTensor.forEach {
            assertTensors(it, mappedActualOutput[it.name] ?: error("Required tensor not found"))
        }
    }
}
