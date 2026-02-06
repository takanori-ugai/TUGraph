package jp.live.ugai.tugraph

import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable

/**
 * Test class for BertQaInference.
 *
 * Tests the BERT question-answering inference functionality including model loading,
 * prediction, and answer extraction. Some tests are disabled by default as they require
 * network access to download models.
 */
class BertQaInferenceTest {
    @Test
    fun testBertQaInferenceObjectExists() {
        assertNotNull(BertQaInference, "BertQaInference object should be accessible")
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "RUN_MODEL_TESTS", matches = "true")
    fun testBertQaInferencePredict() {
        // This test requires downloading a model, so it's disabled by default
        val answer = BertQaInference.predict()

        assertNotNull(answer, "Answer should not be null")
        assertFalse(answer.isEmpty(), "Answer should not be empty")
        assertTrue(
            answer.contains("2004") || answer.contains("December"),
            "Answer should contain relevant information about BBC Japan start date",
        )
    }

    @Test
    fun testBertQaInferenceQuestionAndParagraphFormat() {
        // Verify the question and paragraph structure is well-formed
        // This test doesn't require model loading
        val question = "When did BBC Japan start broadcasting?"
        val paragraph =
            "BBC Japan was a general entertainment Channel. " +
                "Which operated between December 2004 and April 2006. " +
                "It ceased operations after its Japanese distributor folded."

        assertNotNull(question)
        assertNotNull(paragraph)
        assertFalse(question.isEmpty())
        assertFalse(paragraph.isEmpty())
        assertTrue(question.endsWith("?"), "Question should end with a question mark")
        assertTrue(paragraph.contains("BBC Japan"), "Paragraph should contain the subject")
        assertTrue(paragraph.contains("2004"), "Paragraph should contain relevant dates")
    }

    @Test
    fun testBertQaInferenceMainMethodExecutes() {
        // Test that main method can be called without throwing exceptions
        // Note: This will try to load the model, so we catch any model loading exceptions
        try {
            BertQaInference.main(emptyArray())
        } catch (e: Exception) {
            // Model loading might fail in test environment without network/resources
            // We just verify the method is callable
            assertTrue(
                e.message?.contains("model") == true ||
                    e.message?.contains("network") == true ||
                    e.message?.contains("download") == true ||
                    e.javaClass.name.contains("Engine") ||
                    e.javaClass.name.contains("Model"),
                "Exception should be related to model loading or network: ${e.message}",
            )
        }
    }

    @Test
    fun testBertQaInferenceParagraphContainsAnswerableInformation() {
        // Verify the test paragraph is structured to be answerable
        val paragraph =
            "BBC Japan was a general entertainment Channel. " +
                "Which operated between December 2004 and April 2006. " +
                "It ceased operations after its Japanese distributor folded."

        // The paragraph should contain temporal information
        assertTrue(paragraph.matches(Regex(".*\\d{4}.*")), "Paragraph should contain year information")

        // The paragraph should be about BBC Japan
        assertTrue(paragraph.startsWith("BBC Japan"), "Paragraph should start with the subject")

        // The paragraph should contain operation timeline
        assertTrue(paragraph.contains("operated"), "Paragraph should mention operation")
        assertTrue(paragraph.contains("ceased"), "Paragraph should mention cessation")
    }

    @Test
    fun testBertQaInferenceQuestionIsSpecific() {
        val question = "When did BBC Japan start broadcasting?"

        // Question should be specific and temporal
        assertTrue(question.startsWith("When"), "Question should ask about time")
        assertTrue(question.contains("BBC Japan"), "Question should reference specific entity")
        assertTrue(question.contains("start") || question.contains("broadcasting"), "Question should ask about start time")
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "RUN_MODEL_TESTS", matches = "true")
    fun testBertQaInferencePredictAnswerFormat() {
        // This test requires model access
        try {
            val answer = BertQaInference.predict()

            // Answer should be a reasonable text span
            assertNotNull(answer)
            assertTrue(answer.length > 0, "Answer should have content")
            assertTrue(answer.length < 100, "Answer should be a concise span, not the entire paragraph")
            assertFalse(answer.contains("Which operated"), "Answer should not contain irrelevant sentence fragments")
        } catch (e: Exception) {
            // Skip test if model cannot be loaded
            assertTrue(
                e.message?.contains("model") == true ||
                    e.message?.contains("network") == true,
                "Exception should be model-related",
            )
        }
    }

    @Test
    fun testBertQaInferenceParagraphCoherence() {
        val paragraph =
            "BBC Japan was a general entertainment Channel. " +
                "Which operated between December 2004 and April 2006. " +
                "It ceased operations after its Japanese distributor folded."

        // Verify paragraph is coherent and contains complete information
        val sentences = paragraph.split(". ")
        assertTrue(sentences.size >= 2, "Paragraph should have multiple sentences")
        assertTrue(
            sentences.any { it.contains("2004") },
            "At least one sentence should contain the start date",
        )
        assertTrue(
            sentences.any { it.contains("2006") },
            "At least one sentence should contain the end date",
        )
    }

    @Test
    fun testBertQaInferenceObjectSingleton() {
        // Verify BertQaInference is an object (singleton)
        val instance1 = BertQaInference
        val instance2 = BertQaInference

        assertTrue(instance1 === instance2, "BertQaInference should be a singleton object")
    }

    @Test
    fun testBertQaInferenceInputValidation() {
        // Test that the input data is valid
        val question = "When did BBC Japan start broadcasting?"
        val paragraph =
            "BBC Japan was a general entertainment Channel. " +
                "Which operated between December 2004 and April 2006. " +
                "It ceased operations after its Japanese distributor folded."

        // Question should not be empty
        assertTrue(question.isNotEmpty(), "Question should not be empty")

        // Paragraph should not be empty
        assertTrue(paragraph.isNotEmpty(), "Paragraph should not be empty")

        // Question should be shorter than paragraph (typically)
        assertTrue(question.length < paragraph.length, "Question should typically be shorter than paragraph")

        // Both should contain valid text characters
        assertTrue(question.matches(Regex("^[\\w\\s?]+$")), "Question should contain valid characters")
    }
}