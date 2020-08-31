package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class GatherTest {
    private fun getTargetPath(dirName: String) = "/gather/$dirName/"

    @Test
    fun `test gather 0`() {
        Utils.tensorTestRunner(getTargetPath("test_gather_0"))
    }

    @Test
    fun `test gather 1`() {
        Utils.tensorTestRunner(getTargetPath("test_gather_1"))
    }

    @Test
    fun `test gather with negative indices`() {
        Utils.tensorTestRunner(getTargetPath("test_gather_negative_indices"))
    }
}

