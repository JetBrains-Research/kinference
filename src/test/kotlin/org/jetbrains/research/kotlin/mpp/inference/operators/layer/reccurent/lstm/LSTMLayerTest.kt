package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    @Test
    fun `Default test`() {
        val (expectedOutputTensors, actualOutputTensors) = Utils.operatorTestHelper("/lstm_defaults/")

        val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

        for (expectedOutputTensor in expectedOutputTensors){
            val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
            Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
        }
    }

    @Test
    @Suppress("UNCHECKED_CAST")
    fun `BiLSTM test`(){
        val (expectedOutputTensors, actualOutputTensors) = Utils.operatorTestHelper("/bi_lstm_defaults/")

        val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

        for (expectedOutputTensor in expectedOutputTensors){
            val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
            Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
        }
    }
}
