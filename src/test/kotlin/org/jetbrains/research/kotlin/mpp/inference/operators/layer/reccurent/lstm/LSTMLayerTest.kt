package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class LSTMLayerTest {
    @Test
    fun `Default test`() {
        val dataSets = Utils.operatorTestHelper("/lstm/test_lstm_defaults/")
        for (dataSet in dataSets){
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

            for (expectedOutputTensor in expectedOutputTensors){
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
                Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
            }
        }
    }

    @Test
    fun `LSTM with bias`(){
        val dataSets = Utils.operatorTestHelper("/lstm/test_lstm_with_initial_bias/")
        for (dataSet in dataSets){
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

            for (expectedOutputTensor in expectedOutputTensors){
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
                Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
            }
        }
    }

    @Test
    fun `BiLSTM test`(){
        val dataSets = Utils.operatorTestHelper("/lstm/test_bilstm_defaults/")
        for (dataSet in dataSets){
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

            for (expectedOutputTensor in expectedOutputTensors){
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
                Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
            }
        }
    }
}
