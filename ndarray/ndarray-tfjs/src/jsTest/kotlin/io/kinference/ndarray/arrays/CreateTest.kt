package io.kinference.ndarray.arrays

import io.kinference.utils.TestRunner
import kotlin.test.Test

class CreateTest {
    @Test
    fun test_create() = TestRunner.runTest {
        val tensor = NDArrayTFJS.intScalar(300)
        println(tensor.linearSize)
        tensor.close()
    }

}
