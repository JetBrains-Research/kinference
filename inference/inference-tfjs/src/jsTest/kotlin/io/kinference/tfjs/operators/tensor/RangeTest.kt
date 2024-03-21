package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class RangeTest {
    private fun getTargetPath(dirName: String) = "range/$dirName/"

    @Test
    fun test_range_float_type_positive_delta() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_range_float_type_positive_delta"))
    }

    @Test
    fun test_range_int32_type_negative_delta() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_range_int32_type_negative_delta"))
    }
}
