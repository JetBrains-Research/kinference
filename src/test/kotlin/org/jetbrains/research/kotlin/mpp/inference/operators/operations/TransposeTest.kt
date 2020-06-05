package org.jetbrains.research.kotlin.mpp.inference.operators.operations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test
import java.io.File

class TransposeTest {
    @Test
    fun `All tests`(){
        val path = javaClass.getResource("/transpose/").path
        val tests = File(path).list()!!
        for (test in tests) {
            val dataSets = Utils.operatorTestHelper("/transpose/$test/")
            for (dataSet in dataSets) {
                val (expectedOutputTensors, actualOutputTensors) = dataSet

                val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

                for (expectedOutputTensor in expectedOutputTensors){
                    val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
                    Utils.assertTensors(expectedOutputTensor, actualOutputTensor)
                }
            }
        }
    }
}
