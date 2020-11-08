package io.kinference.operators.math

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun `test add`() {
        TestRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun `test add broadcast`() {
        TestRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun `test add scalar`() {
        TestRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
