package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SizeTest {
    private fun getTargetPath(dirName: String) = "size/$dirName/"

    @Test
    fun test_size() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_size"))
    }

    @Test
    fun test_acos_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_size_example"))
    }
}
