package ma.emsi.othmanemoussawi.tp4tests;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;

import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 1 : RAG naïf, avec toutes les étapes du RAG écrites explicitement.
 */
public class RagNaif {

    // Mettre à true pour afficher les segments enregistrés et les segments retrouvés avec leur score.
    private static final boolean AFFICHER_INFOS_RAG = false;

    public static void main(String[] args) {
        configureLogger();
        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Température basse : avec le RAG on veut des réponses fidèles au document.
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ===== Phase 1 du RAG : enregistrement des embeddings =====

        // Parseur PDF fourni par LangChain4j (Apache Tika).
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        // Le fichier est dans src/main/resources, donc à la racine du classpath.
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // Découpage en morceaux de 300 caractères maximum, avec 30 caractères de chevauchement.
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // Modèle d'embeddings local (pas besoin de clé).
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        // embedAll retourne une Response<List<Embedding>>.
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // Magasin d'embeddings en mémoire.
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // ===== Phase 2 du RAG : récupération des segments pertinents =====

        // Les 2 segments les plus pertinents, seulement si leur score est >= 0,5.
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        if (AFFICHER_INFOS_RAG) {
            afficherInfosRag(segments, contentRetriever, embeddingStore, embeddingModel, question);
        }
        System.out.println("Question : " + question);
        System.out.println("Assistant : " + assistant.chat(question));

        conversationAvec(assistant);
    }

    /**
     * Informations pour comprendre les problèmes du RAG (partie optionnelle du test 1).
     */
    private static void afficherInfosRag(List<TextSegment> segments, ContentRetriever contentRetriever,
                                         EmbeddingStore<TextSegment> embeddingStore,
                                         EmbeddingModel embeddingModel, String question) {
        // Segments enregistrés pendant la phase 1.
        System.out.println("===== " + segments.size() + " segments enregistrés =====");
        for (int i = 0; i < segments.size(); i++) {
            System.out.println("--- Segment " + i + " ---");
            System.out.println(segments.get(i).text());
        }

        // Segments retrouvés par le ContentRetriever pendant la phase 2.
        System.out.println("===== Segments retrouvés pour : " + question + " =====");
        List<Content> contents = contentRetriever.retrieve(new Query(question));
        for (Content content : contents) {
            System.out.println("- " + content.textSegment().text());
        }

        // Pour avoir les scores, il faut interroger directement le magasin d'embeddings.
        System.out.println("===== Scores =====");
        Embedding embeddingQuestion = embeddingModel.embed(question).content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddingQuestion)
                .maxResults(2)
                .minScore(0.5)
                .build();
        EmbeddingSearchResult<TextSegment> resultat = embeddingStore.search(request);
        for (EmbeddingMatch<TextSegment> match : resultat.matches()) {
            System.out.println("Score " + match.score() + " : " + match.embedded().text());
        }
    }

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
