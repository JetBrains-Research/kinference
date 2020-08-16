package org.jetbrains.research.kotlin.inference.misc.gpt

import org.jetbrains.research.kotlin.inference.Utils
import org.jetbrains.research.kotlin.inference.model.Model
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.io.File

class GPTTest {
    private fun getTargetPath(dirName: String) = "/gpt/$dirName/"

    @Test
    @Tag("heavy")
    fun `test GPT model`() {
        Utils.tensorTestRunner(getTargetPath("test_dummy_input"))
    }

    @Test
    @Tag("heavy")
    fun `test GPT performance`() {
        val path = object {}.javaClass.getResource("/gpt/test_dummy_input/").path
        val model = Model.load(path + "model.onnx")
        val dataSet = File(path).list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }

            val inputTensors = inputFiles.map { Utils.getTensor(it) }.toList()
            inputTensors
        }.first()

        val count = 100
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
