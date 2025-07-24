package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:standard:en:v1", count = 20)
    }
}

// vectorized
// Test test_data_set_batch1_seq40: avg 92.25, min 82, max 118
// Test test_data_set_batch8_seq40: avg 578.45, min 559, max 599
// Average between inputs: avg 335.35, min 82, max 599


// non vectorized
// Test test_data_set_batch1_seq40: avg 101.95, min 89, max 124
// Test test_data_set_batch8_seq40: avg 647.9, min 613, max 689
// Average between inputs: avg 374.925, min 89, max 689
