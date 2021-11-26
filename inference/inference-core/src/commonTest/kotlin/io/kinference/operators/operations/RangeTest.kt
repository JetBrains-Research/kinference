package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class RangeTest {
    private fun getTargetPath(dirName: String) = "/range/$dirName/"

    @Test
    fun test_range_float_positive_delta() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_range_float_type_positive_delta"))
    }

    @Test
    fun test_range_int_negative_delta() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_range_int32_type_negative_delta"))
    }
}
