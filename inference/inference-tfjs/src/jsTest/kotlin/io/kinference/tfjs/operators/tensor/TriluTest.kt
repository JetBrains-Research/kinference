package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TriluTest {
    private fun getTargetPath(dirName: String) = "trilu/$dirName/"

    @Test
    fun test_tril() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril"))
    }

    @Test
    fun test_tril_neg() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_neg"))
    }

    @Test
    fun test_tril_one_row_neg()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_one_row_neg"))
    }

    @Test
    fun test_tril_out_neg()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_out_neg"))
    }

    @Test
    fun test_tril_out_pos()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_out_pos"))
    }

    @Test
    fun test_tril_pos()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_pos"))
    }

    @Test
    fun test_tril_square()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_square"))
    }

    @Test
    fun test_tril_square_neg()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_square_neg"))
    }

    @Test
    fun test_tril_zero()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_tril_zero"))
    }

    @Test
    fun test_triu() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu"))
    }

    @Test
    fun test_triu_neg() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_neg"))
    }

    @Test
    fun test_triu_one_row()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_one_row"))
    }

    @Test
    fun test_triu_out_neg_out()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_out_neg_out"))
    }

    @Test
    fun test_triu_out_pos()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_out_pos"))
    }

    @Test
    fun test_triu_pos()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_pos"))
    }

    @Test
    fun test_triu_square()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_square"))
    }

    @Test
    fun test_triu_square_neg()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_square_neg"))
    }

    @Test
    fun test_triu_zero()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_triu_zero"))
    }
}
