package io.kinference.ndarray.arrays

import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CreateTest {
    @Test
    fun test_create() = runTest {
        val tensor = NDArrayTFJS.intScalar(300)
        println(tensor.linearSize)
        tensor.close()
    }

}
