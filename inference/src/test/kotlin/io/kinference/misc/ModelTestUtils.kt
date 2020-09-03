package io.kinference.misc

import io.kinference.Utils
import io.kinference.model.Model
import java.io.File

object ModelTestUtils {
    fun testModelPerformance(testDir: String) {
        val path = javaClass.getResource(testDir).path
        val model = Model.load(path + "model.onnx")
        val dataSet = File(path).list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }

            val inputTensors = inputFiles.map { Utils.getTensor(it) }.toList()
            inputTensors
        }.first()

        val count = 10
        val times = LongArray(count)
        for (i in (0 until count)) {
            val startTime = System.currentTimeMillis()
            model.predict(dataSet)
            val endTime = System.currentTimeMillis()
            times[i] = endTime - startTime
        }

        println("Avg: ${times.average()}, min: ${times.min()}, max: ${times.max()}")
    }
}
