package io.kinference.algorithms.completion.model

import io.kinference.algorithms.completion.*
import org.junit.jupiter.api.*
import java.lang.System.currentTimeMillis

class FullModelPassTest {
    companion object {
        private val context  = """
            Kotless is able to deploy existing Spring Boot or Ktor application to AWS serverless platform. To do
            it, you'll need to set up a plugin and replace existing dependency with the appropriate Kotless DSL.

            For **Ktor**, you should replace existing engine (
            e.g. `implementation("io.ktor", "ktor-server-netty", "1.3.2")`)
            with `implementation("io.kotless", "ktor-lang", "0.1.6")`. Note that this dependency bundles Ktor of
            version
            `1.3.2`, so you may need to upgrade other Ktor libraries (like `ktor-html-builder`) to this version.

            For **Spring Boot** you should replace the starter you use (
            e.g. `implementation("org.springframework.boot", "spring-boot-starter-web", "2.3.0.RELASE)`)
            with `implementation("io.kotless", "spring-boot-lang", "0.1.6")`. Note that this dependency bundles
            Spring Boot of version `2.3.0.RELEASE`, so you also may need to upgrade other Spring Boot libraries
            to this version.

            Once it is done, you may hit `deploy` task and
        """.trimIndent()

        private const val prefix = " m"
    }


    @Test
    @Tag("heavy")
    fun `test performance`() {
        val completionModel = CompletionModelFactory.createCompletionModel(CompletionModels.v6)

        performanceTest(completionModel, CompletionModels.v6)
    }

    @Test
    @Tag("heavy")
    fun `test accuracy`() {
        val completionModel = CompletionModelFactory.createCompletionModel(CompletionModels.v6)

        accuracyTest(completionModel, CompletionModels.v6)
    }


    @Test
    @Tag("heavy")
    fun `test bpe problem`() {
        val completionModel = CompletionModelFactory.createCompletionModel(CompletionModels.v6)

        completionModel.complete("Hello my dear friends", "", CompletionModels.v6)
    }

    @Test
    @Tag("heavy")
    fun `test dictionary problem`() {
        val completionModel = CompletionModelFactory.createCompletionModel(CompletionModels.v6)

        val completions = completionModel.complete("Remove old", " modu", CompletionModels.v6)
        Assertions.assertTrue(
            completions.any { it.startsWith(" modules") }
                && completions.all { completion -> completion.all { it.isLetterOrDigit() || it.isWhitespace() || it == ',' } }
        )
    }

    private fun accuracyTest(model: CompletionModel, config: CompletionConfig) {
        val completions = model.complete(context, prefix, config)
        Assertions.assertTrue(completions.any { it.startsWith(" make") })
    }


    private fun performanceTest(model: CompletionModel, config: CompletionConfig) {
        val warmup = 10
        val iterations = 100

        println("Warm up")
        for (i in 0 until warmup) {
            model.complete(context, prefix, config)
        }

        println("Start")
        val startTime = currentTimeMillis()
        for (i in 0 until iterations) {
            model.complete(context, prefix, config)
        }
        val duration = (currentTimeMillis() - startTime) / 1000.0 / iterations
        println(duration)
    }
}
