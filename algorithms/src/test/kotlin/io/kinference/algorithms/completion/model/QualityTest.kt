package io.kinference.algorithms.completion.model

import io.kinference.algorithms.completion.CompletionModelFactory
import io.kinference.algorithms.completion.CompletionModels
import io.kinference.algorithms.completion.evaluation.GolfMetric
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class QualityTest {
    companion object {
        private val text = """
            Kotless stands for Kotlin serverless framework.

            Its focus lies in reducing the routine of serverless deployment creation by generating it straight from the code of the application itself.
            
            So, simply speaking, Kotless gives you one magic button to deploy your Web application as a serverless application on AWS!
            
            Kotless consists of two main parts:
            
            DSL provides a way of defining serverless applications. There are three DSLs supported:
            Kotless DSL — Kotless own DSL that provides annotations to declare routing, scheduled events, etc.
            Ktor — Ktor engine that is introspected by Kotless. Use standard Ktor syntax and Kotless will generate deployment.
            Spring Boot — Spring Boot serverless container that is introspected by Kotless. Use standard Spring syntax and Kotless will generate deployment.
            Kotless Gradle Plugin provides a way of deploying serverless application. For that, it:
            performs the tasks of generating Terraform code from the application code and, subsequently, deploying it to AWS;
            runs application locally, emulates the AWS environment and provides the possibility for in-IDE debugging.
            One of the key features of Kotless is its ability to embed into existing applications. Kotless makes super easy deployment of existing Spring and Ktor applications to AWS serverless platform.
        """.trimIndent()

        private const val prefix = " m"
    }


    @Test
    @Tag("heavy")
    fun `test golf`() {
        val completionModel = CompletionModelFactory.createCompletionModel(CompletionModels.v6)
        val metric = GolfMetric()
        val golfScore = metric.compute(completionModel, text, context_len = 50, CompletionModels.v6)

        println(golfScore)  // 0.6530454895913647
        assert(golfScore < 0.9)
//        ĊĊ

//        0.6507324595219738 - после фильтрации неправильных токенов
    }
}
