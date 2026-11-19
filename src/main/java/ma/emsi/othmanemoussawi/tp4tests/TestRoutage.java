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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3 : routage entre 2 sources ; le LM choisit le ContentRetriever à utiliser.
 */
public class TestRoutage {

    public static void main(String[] args) {
        configureLogger();
        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null) {
            System.out.println("La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ===== Phase 1 : 2 documents enregistrés dans 2 magasins d'embeddings différents =====
        EmbeddingStore<TextSegment> storeIa = ingerer("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> storeCuisine = ingerer("cuisine.pdf", embeddingModel);

        // ===== Phase 2 : un ContentRetriever par magasin =====
        ContentRetriever retrieverIa = creerContentRetriever(storeIa, embeddingModel);
        ContentRetriever retrieverCuisine = creerContentRetriever(storeCuisine, embeddingModel);

        // Description du contenu de chaque source, lue par le LM pour choisir la route.
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverIa,
                "Support de cours sur l'intelligence artificielle : le RAG (Retrieval Augmented Generation), "
                        + "le fine-tuning, les embeddings et LangChain4j");
        descriptions.put(retrieverCuisine,
                "Document sur la cuisine marocaine : plats traditionnels (tajine, couscous, harira, pastilla), "
                        + "recettes, thé à la menthe et pâtisseries");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    /**
     * Phase 1 du RAG pour un fichier : parsing, découpage, embeddings et enregistrement.
     *
     * @param cheminRessource chemin du fichier, relatif à src/main/resources.
     */
    private static EmbeddingStore<TextSegment> ingerer(String cheminRessource, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(cheminRessource, documentParser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
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
