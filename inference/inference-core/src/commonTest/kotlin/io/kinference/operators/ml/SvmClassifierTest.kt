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

    @Test
    fun test_kernel_rbf() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_kernel_rbf"))
    }

    @Test
    fun test_kernel_poly() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_kernel_poly"))
    }

    @Test
    fun test_kernel_sigmoid() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_kernel_sigmoid"))
    }

    @Test
    fun test_transform_softmax() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_transform_softmax"))
    }

    @Test
    fun test_transform_softmax_zero() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_transform_softmax_zero"))
    }

    @Test
    fun test_transform_logistic() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_transform_logistic"))
    }

    @Test
    fun test_transform_probit() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_transform_probit"))
    }

    @Test
    fun test_without_proba_transform_logistic() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_without_proba_transform_logistic"))
    }

    @Test
    fun test_without_proba_no_transform() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_without_proba_no_transform"))
    }

    @Test
    fun test_labels_string() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_labels_string"))
    }

    @Test
    fun test_linear_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_linear_default"))
    }

    @Test
    fun test_linear_one_class() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_linear_one_class"))
    }

    @Test
    fun test_linear_not_all_weights_positive() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_linear_not_all_weights_positive"))
    }

    @Test
    fun test_linear_ten_classes() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_linear_ten_classes"))
    }

    @Test
    fun test_svc_ten_classes() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_svc_ten_classes"))
    }
}
