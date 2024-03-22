package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TriluTest {
    private fun getTargetPath(dirName: String) = "trilu/$dirName/"

    @Test
    fun test_tril() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril"))
    }

    @Test
    fun test_tril_neg() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_neg"))
    }

    @Test
    fun test_tril_one_row_neg()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_one_row_neg"))
    }

    @Test
    fun test_tril_out_neg()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_out_neg"))
    }

    @Test
    fun test_tril_out_pos()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_out_pos"))
    }

    @Test
    fun test_tril_pos()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_pos"))
    }

    @Test
    fun test_tril_square()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_square"))
    }

    @Test
    fun test_tril_square_neg()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_square_neg"))
    }

    @Test
    fun test_tril_zero()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_zero"))
    }

    @Test
    fun test_triu() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu"))
    }

    @Test
    fun test_triu_neg() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_neg"))
    }

    @Test
    fun test_triu_one_row()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_one_row"))
    }

    @Test
    fun test_triu_out_neg_out()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_out_neg_out"))
    }

    @Test
    fun test_triu_out_pos()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_out_pos"))
    }

    @Test
    fun test_triu_pos()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_pos"))
    }

    @Test
    fun test_triu_square()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_square"))
    }

    @Test
    fun test_triu_square_neg()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_square_neg"))
    }

    @Test
    fun test_triu_zero()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_zero"))
    }
}
