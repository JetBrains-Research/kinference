group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))
                api(project(":inference:inference-ort"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-common"))

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.5.2")
                implementation("org.jetbrains.kotlinx:kotlin-deeplearning-dataset:0.5.2")  // Dataset support

                implementation("io.ktor:ktor-client-core:2.3.12")
                implementation("io.ktor:ktor-client-cio:2.3.12") // JVM Engine

                api("org.slf4j:slf4j-api:2.0.9")
                api("org.slf4j:slf4j-simple:2.0.9")

                implementation("ai.djl:api:0.28.0")
                implementation("ai.djl.huggingface:tokenizers:0.28.0")
            }
        }
    }
}
