group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        tasks.withType<Test> {
            useJUnitPlatform {}

            testLogging {
                events("passed", "skipped", "failed")
            }
        }
    }

    js {
        browser {
            testTask {
                useKarma {
                    useChrome()
                }
                testLogging {
                    events("passed", "skipped", "failed")
                }
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))

                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.2")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test-common"))
                implementation(kotlin("test-annotations-common"))

                implementation(project(":utils:test-utils"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api(project(":utils:webgpu-utils:wgpu:jnr-jvm"))
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
                implementation("org.slf4j:slf4j-simple:1.7.30")
                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")
            }
        }
    }
}
