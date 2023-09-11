package io.kinference.operators.ml

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SvmClassifierTest {
    private fun getTargetPath(dirName: String) = "svm_classifier/$dirName/"

    @Test
    fun test_example_0() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_example_0"))
    }

    @Test
    fun test_example_1() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_example_1"))
    }

    @Test
    fun test_example_2() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_example_2"))
    }

    @Test
    fun test_example_3() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_example_3"))
    }
}
