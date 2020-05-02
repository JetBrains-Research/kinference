package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.junit.jupiter.api.Test
import java.io.File

class LSTMLayerTest {
    @Test
    @Suppress("UNCHECKED_CAST")
    fun `Default test`() {
        val path = javaClass.getResource("/lstm_defaults/").path
        val model = Model.load(path + "model.onnx")

        val inputFiles = File(path).walk().filter { "input" in it.name }
        val outputFiles = File(path).walk().filter { "output" in it.name }

        val inputTensors = inputFiles.map { Utils.getTensor(it) }.toList() as List<Tensor<Number>>
        val outputTensors = outputFiles.map { Utils.getTensor(it) }.toList() as List<Tensor<Number>>
        val actualOutputTensors = model.predict(inputTensors) as Collection<Tensor<Number>>
        val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

        for (outputTensor in outputTensors){
            val actualOutputTensor = mappedActualOutputTensors[outputTensor.name] ?: error("Required tensor not found")
            Utils.assertTensors(outputTensor, actualOutputTensor)
        }
    }
}
