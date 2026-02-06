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
    fun testBertQaInferenceStaticInputsWellFormed() {
        val question = BertQaInference.QUESTION
        val paragraph = BertQaInference.PARAGRAPH

        assertNotNull(question)
        assertNotNull(paragraph)
        assertFalse(question.isEmpty())
        assertFalse(paragraph.isEmpty())
        assertTrue(question.endsWith("?"), "Question should end with a question mark")
        assertTrue(question.startsWith("When"), "Question should ask about time")
        assertTrue(question.contains("BBC Japan"), "Question should reference specific entity")
        assertTrue(question.contains("start") || question.contains("broadcasting"), "Question should ask about start time")
        assertTrue(paragraph.startsWith("BBC Japan"), "Paragraph should start with the subject")
        assertTrue(paragraph.contains("operated"), "Paragraph should mention operation")
        assertTrue(paragraph.contains("ceased"), "Paragraph should mention cessation")
        assertTrue(paragraph.matches(Regex(".*\\d{4}.*")), "Paragraph should contain year information")
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "RUN_MODEL_TESTS", matches = "true")
    fun testBertQaInferenceMainMethodExecutes() {
        // With model available, main should complete without exception
        BertQaInference.main(emptyArray())
    }

    @Test
    @EnabledIfEnvironmentVariable(named = "RUN_MODEL_TESTS", matches = "true")
    fun testBertQaInferencePredictAnswerFormat() {
        val answer = BertQaInference.predict()

        assertNotNull(answer)
        assertTrue(answer.length > 0, "Answer should have content")
        assertTrue(answer.length < 100, "Answer should be a concise span, not the entire paragraph")
        assertFalse(answer.contains("Which operated"), "Answer should not contain irrelevant sentence fragments")
    }
}
