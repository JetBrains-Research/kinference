import org.jetbrains.kotlin.gradle.dsl.KotlinCompile

group = rootProject.group
version = rootProject.version

kotlin {
    jvm()
    js() {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            repositories {
                mavenCentral()
                mavenLocal()
            }

            dependencies {
                api(kotlin("stdlib"))
                api("io.kinference.primitives:primitives-annotations:0.1.8")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
            }
        }

        val jvmMain by getting {

        }
    }
}


dependencies {
    kotlinCompilerPluginClasspath("io.kinference.primitives", "kotlin-plugin", "0.1.8")
}

val generatedDir = "$projectDir/src/commonMain/kotlin-gen"
val incrementalDir = "$buildDir/"

tasks.withType<KotlinCompile<*>> {
    kotlinOptions {
        freeCompilerArgs = freeCompilerArgs + listOf(
            "-P",
            "plugin:io.kinference.primitives.kotlin-plugin:outputDir=$generatedDir",
            "-P",
            "plugin:io.kinference.primitives.kotlin-plugin:icOutputDir=$incrementalDir"
        )
    }
}

tasks["compileKotlinJs"].dependsOn("compileKotlinJvm")

kotlin {
    sourceSets["commonMain"].apply {
        kotlin.srcDirs(generatedDir)
    }
}
