package io.kinference.generating

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class ModelWrapperTest {
    @Test
    @Tag("heavy")
    fun testExecutable() {
        val baseDir = "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/src"
        val model = OnnxModelWrapper("$baseDir/completion/big/opt/onnxrt/onnx_models/distilgpt2_opt.onnx")
        val res = model.initLastLogProbs(listOf(1, 2, 3, 452))
        print(res)
        assertTrue(true)
    }
}
