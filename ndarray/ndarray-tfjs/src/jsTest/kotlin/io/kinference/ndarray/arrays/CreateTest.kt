package io.kinference.ndarray.arrays

import kotlin.test.Test

class CreateTest {
    @Test
    fun test_create() {
        val tensor = NDArrayTFJS.intScalar(300)
        println(tensor.linearSize)
        tensor.close()
    }

}
