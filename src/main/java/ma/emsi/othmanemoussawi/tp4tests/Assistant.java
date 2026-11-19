package ma.emsi.othmanemoussawi.tp4tests;

/**
 * Assistant conversationnel utilisé par tous les tests.
 * L'implémentation est fournie par AiServices de LangChain4j.
 */
public interface Assistant {
    // Prend un message de l'utilisateur et retourne une réponse du LLM.
    String chat(String userMessage);
}
