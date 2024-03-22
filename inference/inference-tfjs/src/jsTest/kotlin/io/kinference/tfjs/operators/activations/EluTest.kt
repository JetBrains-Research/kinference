package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class EluTest {
    private fun getTargetPath(dirName: String) = "elu/$dirName/"

    @Test
    fun test_elu() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu"))
    }

    @Test
    fun test_elu_default() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu_default"))
    }

    @Test
    fun test_elu_default_expanded_ver18() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu_default_expanded_ver18"))
    }

    @Test
    fun test_elu_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu_example"))
    }

    @Test
    fun test_elu_example_expanded_ver18() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu_example_expanded_ver18"))
    }

    @Test
    fun test_elu_expanded_ver18() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_elu_expanded_ver18"))
    }
}
